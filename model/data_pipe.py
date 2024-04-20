from fastNLP.io import ConllLoader, Loader
from fastNLP.io.loader.conll import _read_conll
from fastNLP.io.pipe.utils import iob2, iob2bioes
from fastNLP import DataSet, Instance
from fastNLP.io import Pipe
from transformers import AutoTokenizer
from fastNLP.core.metrics import _bio_tag_to_spans
from fastNLP.io import DataBundle
import numpy as np
from itertools import chain
from fastNLP import Const
from functools import cmp_to_key
import json
from copy import deepcopy
from tqdm import tqdm
import os
import torch
import torchvision
from fastNLP import cache_results


class BartNERPipe(Pipe):
    def __init__(self,image_feature_path=None, 
                      image_annotation_path= None,
                      max_bbox =16,
                      normalize=False,
                      tokenizer='facebook/bart-base', 
                      target_type='word',region_dim=2048):
        """

        :param tokenizer:
        :param dataset_name:
        :param target_type:
            word: 生成word的start; #仅支持
            bpe: 生成所有的bpe
            span: 每一段按照start end生成
            span_bpe: 每一段都是start的所有bpe，end的所有bpe
        """
        super().__init__()
        
        
        self.image_feature_path=image_feature_path  # vinvl 使用
        self.image_annotation_path = image_annotation_path
        
        self.max_bbox= max_bbox  
        self.max_aspect = 6
        self.region_dim=region_dim #2048 # debug
        self.normalize = normalize

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        
        assert target_type in ('word') #

        cur_num_tokens = self.tokenizer.vocab_size
        self.num_token_in_orig_tokenizer = cur_num_tokens
        self.target_type = target_type

        
        self.not_cover = 0

    def add_tags_to_special_tokens(self, data_bundle):
        
        mapping ={}
        
        ## targt-region relation
        mapping['0'] ='<<which region>>'  ## 取自 region_label 的最后一个item,相关是0 不相关是1
        mapping['1'] ='<<no region>>'
        
        ## 4 entity-type 
        mapping['loc'] = '<<location>>'
        mapping['per'] = '<<person>>'
        mapping['other'] = '<<others>>'
        mapping['org'] = '<<organization>>'
        self.mapping =mapping

        sorted_add_tokens = list(mapping.values())
        unique_no_split_tokens = self.tokenizer.unique_no_split_tokens
        for tok in sorted_add_tokens:
            assert self.tokenizer.convert_tokens_to_ids([tok])[0] == self.tokenizer.unk_token_id
        self.tokenizer.unique_no_split_tokens = unique_no_split_tokens + sorted_add_tokens
        self.tokenizer.add_tokens(sorted_add_tokens)
        self.mapping2id = {}  # 给定转换后的tag，输出的是在tokenizer中的id，用来初始化表示
        self.mapping2targetid = {}  # 给定原始tag，输出对应的数字

        for key, value in self.mapping.items():
            key_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(value))
            assert len(key_id) == 1, value
            assert key_id[0] >= self.num_token_in_orig_tokenizer
            self.mapping2id[value] = key_id[0]  
            self.mapping2targetid[key] = len(self.mapping2targetid)

    def read_image_label(self,img_id):
        import xml.etree.ElementTree as ET
        fn=os.path.join(self.image_annotation_path,img_id+'.xml')
        tree=ET.parse(fn)
        root=tree.getroot()
        aspects = []
        boxes = []
        for object_container in root.findall('object'):
            for names in object_container.findall('name'):
                box_name=names.text
                box_container = object_container.findall('bndbox')
                if len(box_container) > 0:
                    xmin = int(box_container[0].findall('xmin')[0].text) 
                    ymin = int(box_container[0].findall('ymin')[0].text) 
                    xmax = int(box_container[0].findall('xmax')[0].text) 
                    ymax = int(box_container[0].findall('ymax')[0].text) 
                aspects.append(box_name)
                boxes.append([xmin,ymin,xmax,ymax])
        return aspects, boxes


    def process_new(self, data_bundle):
        
        self.add_tags_to_special_tokens(data_bundle)
        # 转换tag
        target_shift = len(self.mapping) + 2  # {'0': '<<which region>>', '1': '<<no region>>', 'loc': '<<location>>',
        # 'per': '<<person>>', 'other': '<<others>>', 'org': '<<organization>>'}

        def prepare_target(ins):
            img_id = ins['img_id']
            
            image_num = 0
            image_tag =''
            image_boxes = np.zeros((self.max_bbox,4),dtype= np.float32)
            image_feature = np.zeros((self.max_bbox, self.region_dim),dtype= np.float32)

            relations= np.zeros((50, 2),dtype= np.int8) #self.max_rel=50
            if self.image_feature_path:
                ########### Vinvl image feature
                try:
                    # import pdb;pdb.set_trace()
                    obj_exist = os.path.exists(os.path.join(self.image_feature_path,str(img_id)+'.jpg.npz'))
                    rel_num = 0
                    rel_num_final=0
                    if obj_exist:
                        # print(img_id)
                        # if img_id== '16_05_16_396': # '20175': #'71044':
                        #     print(img_id)
                        #     import pdb;pdb.set_trace()
                        img=np.load(os.path.join(self.image_feature_path,str(img_id)+'.jpg.npz'),allow_pickle=True) # .jpg.npz
                        image_num = img['num_boxes']
                        image_feature_ = img['box_features']
                        if self.normalize:
                            image_feature_ = (image_feature_/np.sqrt((image_feature_**2).sum()))
                        final_num = min(image_num,self.max_bbox)
                        image_feature[:final_num] = image_feature_[:final_num]
                        image_boxes[:final_num] = img['bounding_boxes'][:final_num]
                        # ['num_boxes', 'image_h', 'image_w', 'bounding_boxes', 'box_features', 'scores', 'objects', 'attr_obj', 'attr_scores', 'scores_all']
                        # img_file_name=os.path.join('../data/IJCAI2019_data/twitter2015_images/',str(img_id)+'.jpg')
                        object_names=img['objects'][:final_num]
                        object_scores=img['scores'][:final_num]
                        # attr_names=img['attr_obj']


                        # if not np.all(img['relations'] == None):
                        rel_num = len(img['relations'])
                        

                        if rel_num>0: # len(img['relations'].shape)>0:
                            

                            # relations=img['relations']
                            # relation_scores = img['relation_scores']
                            # relation_predicates = img['relation_predicates']
                            # relation_texts = img['relation_texts']
                            # import pdb;pdb.set_trace()

                            select_indexes=[] #先过滤，再筛选有效的关系。
                            for i,t in enumerate(img['relations']):
                                s,e=t # img['relations']
                                if s<final_num and e<final_num:
                                    select_indexes.append(i)
                                


                            rel_num_final=min(len(select_indexes),50)
                            relations[:rel_num_final]=img['relations'][select_indexes][:rel_num_final]
                            relation_scores = img['relation_scores'][select_indexes][:rel_num_final]
                            relation_predicates = img['relation_predicates'][select_indexes][:rel_num_final]
                            relation_texts = img['relation_texts'][select_indexes][:rel_num_final]
                        else:
                            # relations=None
                            print('hh')

                    else:
                        print('not exists')

                except:
                    print("no image feature"+str(img_id)) 
            else:
                print("image feature error!")
            obj_bpes = []
            rel_bpes = []
            trip_bpes = []
            # rel_trip=[]
            if obj_exist:
                for obj in object_names:
                    obj_txt = self.tokenizer.tokenize(obj, add_prefix_space=True)
                    obj_txt = self.tokenizer.convert_tokens_to_ids(obj_txt)
                    obj_bpes.append(obj_txt)
                # if img_id=='O_4193':# 966333 71044 
                #     import pdb;pdb.set_trace()
                #     print(img_id)
                if rel_num_final>0:
                    # import pdb;pdb.set_trace()
                    # rel_txt_total=''
                    for i,rel in enumerate(relation_texts):
                        start,end=relations[i]
                        # # v0
                        # labels_hv_idx = [str(x)+"-"+str(i) for i, x in enumerate(object_names)]
                        # rel_txt+=labels_hv_idx[start] +" " + relation_texts[i] +" " + labels_hv_idx[end]
                        # # v1
                        # print(i)
                        # import pdb;pdb.set_trace()
                        rel_txt=object_names[start] +" " + relation_texts[i] +" " + object_names[end]+","
                        # rel_txt_total+=rel_txt
                        rel = self.tokenizer.tokenize(rel, add_prefix_space=True)
                        rel = self.tokenizer.convert_tokens_to_ids(rel)
                        trip_bpe = self.tokenizer.tokenize(rel_txt, add_prefix_space=True)
                        trip_bpe = self.tokenizer.convert_tokens_to_ids(trip_bpe)
                        rel_bpes.append(rel)
                        trip_bpes.append(trip_bpe)
                    # rel_txt_total = self.tokenizer.tokenize(rel_txt_total, add_prefix_space=True)
                    # rel_trip = self.tokenizer.convert_tokens_to_ids(rel_txt_total)
                    # import pdb;pdb.set_trace()
                    
                    
                else:
                    # print(img_id)
                    rel_bpes=[0] # for padding, 防报错
                    # rel_trip=[0]
                    trip_bpes=[0]
            else:
                rel_bpes=[0] #greatte
                # rel_trip=[0]
                # obj_bpes=[0]
                trip_bpes=[0]

            raw_words = ins['raw_words']
            word_bpes = [[self.tokenizer.bos_token_id]]
            first = []  # 用来取每个word第一个bpe
            cur_bpe_len = 1
            for word in raw_words:
                bpes = self.tokenizer.tokenize(word, add_prefix_space=True)
                bpes = self.tokenizer.convert_tokens_to_ids(bpes)
                first.append(cur_bpe_len)
                cur_bpe_len += len(bpes)
                word_bpes.append(bpes)
            assert first[-1] + len(bpes) == sum(map(len, word_bpes))
            word_bpes.append([self.tokenizer.eos_token_id])
            assert len(first) == len(raw_words) == len(word_bpes) - 2   ## raw_word

            lens = list(map(len, word_bpes))
            cum_lens = np.cumsum(lens).tolist()
            first = list(range(cum_lens[-1]))  ## first 只掩码sentence内部的?? 所以搞这么多，first是用来干啥的



            ###### image_label #######
            aspect_ious_dic ={}   ## {aspect:[iou1,iou2,...]}
            if os.path.exists(os.path.join(self.image_annotation_path,img_id+'.xml')): #
                names, gt_boxes = self.read_image_label(img_id)
                assert len(names) > 0, img_id #每个image确保至少一个实体?
                IoUs=(torchvision.ops.box_iou(torch.tensor(gt_boxes),torch.tensor(image_boxes))).numpy() #[x,4],[16,4]  ->[x,16]
                
                for i,nn in enumerate(names):  ## 对于每个标注框
                    cur_iou = IoUs[i]
                    if max(cur_iou) < 0.5: ## object detector 没有检测到
                        self.not_cover +=1  ## ps: not_cover 这个数量是针对每个标注框的，不是每个实体的。
                        if nn not in aspect_ious_dic: ## 首次出现这个name，赋 -1；如果之前已经有这个name的记录，此处都不再关注
                            aspect_ious_dic[nn] = np.array([-1])  
                    else:
                        if nn in aspect_ious_dic:  ## 如果一个name(aspect)对应多个标注框，更新iou
                            last_iou = aspect_ious_dic[nn]# 上一个该name对应的iou
                            if last_iou[0] == -1:  
                                aspect_ious_dic[nn] = cur_iou ## 直接赋当前iou
                            else: ## 该aspect有多个标注框，且多个框被检测到
                                final_iou = np.array([max(last_iou[i],cur_iou[i]) for i in range(len(last_iou))]) #对于每个预测，取两次中最大iou值。遍历所有预测。
                                aspect_ious_dic[nn] = final_iou
                        else: ## 第一次碰到该name，直接添加。一般情况，一个name中，只有一个标注框。
                            aspect_ious_dic[nn] = cur_iou
            
            
           
            region_label = []
            cover_flag = []  ## 0:entity-region 相关，但 detector 没有检测到 ；1: entity-region 相关，且检测到；2:entity-region 不相关
            entities = ins['entities']  # [[ent1, ent2,], [ent1, ent2]]
            for i,e in enumerate(entities):
                e = ' '.join(e)
                if e in aspect_ious_dic: # e是从文本的实体标注中取出的。if True，标注了。
                    ori_ious = aspect_ious_dic[e]
                    ### 处理notcover
                    if ori_ious[0] == -1:
                        average_iou = 0.
                        region_label.append(np.array([average_iou]*self.max_bbox + [1.]))   ## 按照不相关训练
                        cover_flag.append(np.array([0])) ## 按照相关评估 ## 
                    else: #标注了，但VinVL未检出（没有一个检测框和e的GT之IoU大于0.5）
                        keeped_ious = np.array([iou if iou >0.5 else 0 for iou in ori_ious])
                        norm_iou = keeped_ious / float(sum(keeped_ious))
                        region_label.append(np.append(norm_iou,[0.])) # 续了一位是未检出位，1代表属于这一类。
                        cover_flag.append(np.array([1]))
                    
                else: #没标注这个实体。
                    average_iou = 0.  # 0. # 1 / self.max_bbox
                    region_label.append(np.array([average_iou]*self.max_bbox + [1.]))
                    cover_flag.append(np.array([2]))
            
            ## 全 O 是 [], 会报错，先pad一个
            if len(region_label) ==0:
                region_label.append(np.array([0.]*(self.max_bbox +1)))
            if len(cover_flag) ==0:
                cover_flag.append(np.array([2]))
            


            # constructing target:(the output)
            entity_spans = ins['entity_spans']  # [(s1, e1, s2, e2), ()]
            entity_tags = ins['entity_tags']  # [tag1, tag2...]
            target = [0]  
            pairs = [] # [[9, 10, 3, 7], [16, 3, 4]]
            
            assert len(entity_spans) == len(entity_tags)
            _word_bpes = list(chain(*word_bpes))
            for idx, (entity, tag) in enumerate(zip(entity_spans, entity_tags)): # for each entity
                cur_pair = []
                num_ent = len(entity) // 2
                assert num_ent ==1 # 新加。证明下面这个for循环是没必要的，猜测是为了处理不连续实体？
                for i in range(num_ent):#以下的并非多余，因为分词后，一个词变多个词，start和end的位置会发生变化。 # cur_pair_==[1, 2] , target_shift: len(mapping)==8;-->cur_pair: [9, 10]
                    start = entity[2 * i]
                    end = entity[2 * i + 1]
                    cur_pair_ = []
                    if self.target_type == 'word':
                        cur_pair_.extend([cum_lens[k] for k in list(range(start, end))])
                    elif self.target_type == 'span':
                        cur_pair_.append(cum_lens[start])
                        cur_pair_.append(cum_lens[end]-1)  # it is more reasonable to use ``cur_pair_.append(cum_lens[end-1])``
                    elif self.target_type == 'span_bpe':
                        cur_pair_.extend(
                            list(range(cum_lens[start], cum_lens[start + 1])))  # 由于cum_lens是[1, 3...]即第0位其实就是cls之后的了
                        cur_pair_.extend(
                            list(range(cum_lens[end - 1], cum_lens[end])))  # 由于cum_lens是[1, 3...]即第0位其实就是cls之后的了
                    elif self.target_type == 'bpe':
                        cur_pair_.extend(list(range(cum_lens[start], cum_lens[end])))
                    else:
                        raise RuntimeError("Not support other tagging")
                    cur_pair.extend([p + target_shift for p in cur_pair_])
                for _, (j, word_idx) in enumerate(zip((cur_pair[0], cur_pair[-1]), (0, -1))):
                    j = j - target_shift
                    if 'word' == self.target_type or word_idx != -1:
                        assert _word_bpes[j] == \
                               self.tokenizer.convert_tokens_to_ids(
                                   self.tokenizer.tokenize(entities[idx][word_idx], add_prefix_space=True)[:1])[0]
                    else:
                        assert _word_bpes[j] == \
                               self.tokenizer.convert_tokens_to_ids(
                                   self.tokenizer.tokenize(entities[idx][word_idx], add_prefix_space=True)[-1:])[0]
                assert all([cur_pair[i] < cum_lens[-1] + target_shift for i in range(len(cur_pair))])
                               
                cur_pair.append(self.mapping2targetid[str(int(region_label[idx][-1]))] +2)  ##  entity-region relation
                cur_pair.append(self.mapping2targetid[tag] + 2)  # 加2是由于有shift ... # [9, 10, 11, 12, 14, 15, 3, 6]
                ### ↑ [span, <<which region>>(+Linear), type] / [span, <<no region>>, type]       [9, 10, 3, 7],[span:,9,10],<<3=1+2>>,tag
                
                pairs.append([p for p in cur_pair])
                
            target.extend(list(chain(*pairs)))
            target.append(1)  # 特殊的eos

            word_bpes = list(chain(*word_bpes))
            obj_lens=[0]
            if obj_exist:
                obj_lens=[len(obj) for obj in obj_bpes]
                obj_bpes=list(chain(*obj_bpes))

            rel_lens=[0]
            trip_lens=[0]
            # import pdb;pdb.set_trace()
            if rel_num_final>0:
                rel_lens=[len(rel) for rel in rel_bpes] # dict: 'rel_lens':rel_lens,
                rel_bpes=list(chain(*rel_bpes))
                trip_lens=[len(rel) for rel in trip_bpes]
                trip_bpes=list(chain(*trip_bpes))
                if rel_bpes==[]:
                    import pdb;pdb.set_trace()
                    print('strange!!')
            
            # if rel_lens[0]>=1 and (relations[0]==0).all():
            #     import pdb;pdb.set_trace()
            #     print('strange!!')
            # if len(np.where(np.array(rel_lens) == 0)[0]) > 0:
            #     print("error")
            dict  = {'tgt_tokens': target, 'target_span': pairs, 'src_tokens': word_bpes,
                    'first': first,'image_tag':image_tag, 'image_feature':image_feature,'region_label':region_label,'cover_flag':cover_flag,'obj_bpes':obj_bpes,'obj_lens':obj_lens,'relations':relations,'rel_bpes':rel_bpes,'rel_lens':rel_lens,'rel_trips':trip_bpes,'trip_lens':trip_lens}
            # 'rel_num':[rel_num],
            # else:
            #     dict = {'tgt_tokens': target, 'target_span': pairs, 'src_tokens': word_bpes,
            #             'first': first, 'image_tag': image_tag, 'image_feature': image_feature,
            #             'region_label': region_label, 'cover_flag': cover_flag, 'obj_bpes': [],'obj_lens': [0], 'relations':[],'rel_num':[0],'rel_bpes': [], }

            return dict
        # for name in list(data_bundle.datasets.keys()):
        #     data_bundle.datasets[name]=data_bundle.datasets[name][:1]

        data_bundle.apply_more(prepare_target, use_tqdm=False, tqdm_desc='pre. tgt.')  

        data_bundle.set_ignore_type('target_span', 'entities') 
        data_bundle.set_ignore_type('image_tag')
        data_bundle.set_pad_val('tgt_tokens', 1)  # 设置为eos所在的id
        data_bundle.set_pad_val('src_tokens', self.tokenizer.pad_token_id)

        data_bundle.apply_field(lambda x: len(x), field_name='src_tokens', new_field_name='src_seq_len')
        # data_bundle.apply_field(lambda x: len(x), field_name='rel_trips', new_field_name='trip_lens')
        data_bundle.apply_field(lambda x: len(x), field_name='tgt_tokens', new_field_name='tgt_seq_len')
        # data_bundle.set_input('tgt_tokens', 'src_tokens', 'src_seq_len', 'tgt_seq_len', 'first','image_feature','image_tag')
        data_bundle.set_target('tgt_tokens', 'tgt_seq_len', 'target_span', 'entities','region_label','cover_flag')
        data_bundle.set_input('tgt_tokens', 'src_tokens', 'src_seq_len', 'tgt_seq_len', 'first','image_feature','image_tag','obj_bpes','obj_lens','relations','rel_bpes','rel_lens','rel_trips','trip_lens') #
        print("not_cover: %d"%(self.not_cover))
        return data_bundle



    def process_from_file(self, paths, demo=False) -> DataBundle:

        # 读取数据
        if isinstance(paths, str):
            path = paths
        else:
            path = paths['train']
        
        data_bundle = TwitterNer(demo=demo).load(paths)
        
        data_bundle = self.process_new(data_bundle)
        
        return data_bundle



class TwitterNer(ConllLoader):
   

    def __init__(self, demo=False):
        headers = [
            'raw_words', 'target',
        ]
        # most of the data should put the label in the last column.
        super().__init__(headers=headers, indexes=[0, -1])
        self.demo = demo
    def read_file(self,filename):
        
        f=open(filename)
        data=[]
        raw_data=[]
        target=[]
        coarse_target=[]
        for line in f:
            if len(line)==0 or line.startswith('-DOCSTART') or line[0]=="\n":
                if len(raw_data) > 0:
                    #import pdb;pdb.set_trace()
                    data.append((raw_data,target, coarse_target))
                    raw_data = []
                    target = []
                    coarse_target=[]
                continue
            splits = line.split('\t')
            if len(splits) == 1: ## Imageid
                raw_data.append(splits[0][:-1])
            else:
                raw_data.append(splits[0])
                target.append(splits[-1][:-1]) 
                coarse_target.append("O")
        if len(raw_data) >0:
            data.append((raw_data,target, coarse_target))
            raw_data = []
            target = []
            coarse_target=[]
        print("The number of samples: "+ str(len(data)))
        return data


    def _load(self, path):
        
        ds = DataSet()
        data = self.read_file(path)
        for raw_words, target, coarse_target in data:
            img_id = raw_words[0][6:]
            raw_words = raw_words[1:]  #去除第一个token raw_words[0]='IMGID:XXX
            target = iob2(target)      #同上
            spans = _bio_tag_to_spans(target)  #Example:('person_other', (8, 10))  #从0开始
            coarse_target = iob2(coarse_target)
            entities = []
            entity_tags = []
            entity_spans = []
            for tag, (start, end) in spans:
                entities.append(raw_words[start:end])
                entity_tags.append(tag.lower())
                entity_spans.append([start, end])

            ds.append(Instance(img_id=img_id, raw_words=raw_words, entities=entities, entity_tags=entity_tags,  
                               entity_spans=entity_spans, raw_target=target, coarse_target=coarse_target))
            if self.demo and len(ds) > 30:
                break
        if len(ds) == 0:
            raise RuntimeError("No data found {}.".format(path))
        return ds





if __name__ == '__main__':
    data_bundle = TwitterNer(demo=False).load('data/twitter')
    BartNERPipe(target_type='word', dataset_name='twitter').process(data_bundle)
