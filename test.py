import sys
sys.path.append('../')
import os
if 'p' in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = os.environ['p']
    # os.environ['CUDA_VISIBLE_DEVICES'] = '7'

import warnings
                                                 
warnings.filterwarnings('ignore')


from model.data_pipe import BartNERPipe
from model.bart_multi_concat import BartSeq2SeqModel  
from model.generater_multi_concat import SequenceGeneratorModel     
from model.metrics import Seq2SeqSpanMetric 
from model.losses import get_loss

import fitlog
import datetime
from fastNLP import Trainer

from torch import optim
from fastNLP import BucketSampler, GradientClipCallback, cache_results, EarlyStopCallback, SequentialSampler

from model.callbacks import WarmupCallback
from fastNLP.core.sampler import SortedSampler
# from fastNLP.core.sampler import  ConstTokenNumSampler
from model.callbacks import FitlogCallback
from fastNLP import DataSetIter
from tqdm import tqdm, trange
from fastNLP.core.utils import _move_dict_value_to_device
import random
import torch
import numpy as np

fitlog.debug()
fitlog.set_log_dir('logs')




import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--bart_name', default='facebook/bart-large', type=str)
parser.add_argument('--datapath', default='./Twitter_GMNER/txt/', type=str)
parser.add_argument('--image_feature_path',default='./data/Twitter_GMNER_vinvl', type=str)
parser.add_argument('--image_annotation_path',default='./Twitter_GMNER/xml/', type=str)
parser.add_argument('--box_num',default='16', type=int)
parser.add_argument('--model_weight',default= None, type = str)
parser.add_argument('--normalize',default=False, action = "store_true")
parser.add_argument('--max_len', default=30, type=int)
parser.add_argument('--batch_size',default=16,type=int)
parser.add_argument("--log",default='./logs',type=str)
parser.add_argument('--device', default="cuda:0") #'cuda:0'
args= parser.parse_args()


model_path = args.model_weight.rsplit('/')
args.pred_output_file = '/'.join(model_path[:-1])+'/total-pred_'+model_path[-1]+'.txt'
refresh_data=False

dataset_name = 'twitter-ner'
args.length_penalty = 1
args.target_type = 'word'
args.schedule = 'linear'
args.decoder_type = 'avg_feature'
args.num_beams = 1   
args.use_encoder_mlp = 1
args.warmup_ratio = 0.01

device = args.device

eval_start_epoch = 0


if 'twitter' in dataset_name:  
    max_len, max_len_a = args.max_len, 0.6
else:
    print("Error dataset_name!")


if isinstance(args.decoder_type, str) and args.decoder_type.lower() == 'none':
    args.decoder_type = None
demo = False

if '15' in args.datapath:
    cache_name='cache/twitter15'
elif '17' in args.datapath:
    cache_name='cache/twitter17'
else:
    cache_name='cache/gmner'
@cache_results(_cache_fp=cache_name, _refresh=refresh_data)
def get_data():

    pipe = BartNERPipe(image_feature_path=args.image_feature_path, 
                       image_annotation_path=args.image_annotation_path,
                       max_bbox =args.box_num,
                       normalize=args.normalize,
                       tokenizer=args.bart_name, 
                       target_type=args.target_type)
    if dataset_name == 'twitter-ner': 
        paths ={
            'train': os.path.join(args.datapath,'train.txt'),
            'dev': os.path.join(args.datapath,'dev.txt'),
            'test': os.path.join(args.datapath,'test.txt') }

        data_bundle = pipe.process_from_file(paths, demo=demo)
        
    return data_bundle, pipe.tokenizer, pipe.mapping2id

data_bundle, tokenizer, mapping2id = get_data()

print(f'max_len_a:{max_len_a}, max_len:{max_len}')

print(data_bundle)
print("The number of tokens in tokenizer ", len(tokenizer.decoder))  

bos_token_id = 0
eos_token_id = 1
label_ids = list(mapping2id.values())


model = BartSeq2SeqModel.build_model(args.bart_name, tokenizer, label_ids=label_ids, decoder_type=args.decoder_type,
                                     use_encoder_mlp=args.use_encoder_mlp,box_num = args.box_num)


vocab_size = len(tokenizer)

model = SequenceGeneratorModel(model, bos_token_id=bos_token_id,
                               eos_token_id=eos_token_id, 
                               max_length=max_len, max_len_a=max_len_a,num_beams=args.num_beams, do_sample=False,
                               repetition_penalty=1, length_penalty=args.length_penalty, pad_token_id=eos_token_id,
                               restricter=None, top_k = 1
                               )

model.load_state_dict(torch.load(args.model_weight, map_location=device))

import torch
if device is None:
    if torch.cuda.is_available():
            device = 'cuda'
    else:
        device = 'cpu'
print('device: '+str(device) + '\n')



metric = Seq2SeqSpanMetric(eos_token_id, num_labels=len(label_ids), region_num =args.box_num, target_type=args.target_type,print_mode = False )


test_dataset = data_bundle.get_dataset('test')
print(test_dataset[:3])

test_dataset.set_target('raw_words', 'raw_target')




# test_analysis_saved_path = "./saved_test_analysis/61.3/dev"

device = torch.device(device)
model.to(device)

def Predict(args,eval_data, model, device, metric,tokenizer,ids2label):
    data_iterator = DataSetIter(eval_data, batch_size=args.batch_size, sampler=SequentialSampler())
    # for batch_x, batch_y in tqdm(data_iterator, total=len(data_iterator)):
    with open (args.pred_output_file,'w') as fw:
        for batch_idx, (batch_x, batch_y) in enumerate(data_iterator):
            _move_dict_value_to_device(batch_x, batch_y, device=device)
            src_tokens = batch_x['src_tokens']
            image_feature = batch_x['image_feature']
            tgt_tokens = batch_x['tgt_tokens']
            src_seq_len = batch_x['src_seq_len']
            tgt_seq_len = batch_x['tgt_seq_len']
            first = batch_x['first']
            object_names=batch_x['obj_bpes']
            obj_lens = batch_x['obj_lens']
            region_label = batch_y['region_label']
            target_span = batch_y['target_span']
            cover_flag = batch_y['cover_flag']
            
            rel_adjs=batch_x['relations']
            rel_lens=batch_x['rel_lens']
            rel_trips=batch_x['rel_trips']
            trip_lens=batch_x['trip_lens']
            # import pdb;pdb.set_trace()
            results = model.predict(src_tokens,image_feature, src_seq_len=src_seq_len, first=first,object_names=object_names,obj_lens=obj_lens,rel_adjs=rel_adjs,rel_lens=rel_lens,rel_trips=rel_trips,trip_lens=trip_lens)
            pred,region_pred = results['pred'],results['region_pred']   ## logits:(bsz,tgt_len,class+max_len)  region_logits:(??,8)
            test_analysis = results["test_analysis"]

            pred_pairs, target_pairs = metric.evaluate(target_span, pred, tgt_tokens, region_pred,region_label,cover_flag,predict_mode=True)
            
            raw_words = batch_y['raw_words']
            word_start_index = 8 ## 2 + 2 +4
            assert len(pred_pairs) == len(target_pairs)
            for i in range(len(pred_pairs)):
                input_ids = test_analysis["input_ids"][i]
                txt_gat_attentions = test_analysis["txt_gat_attentions"][:, i]
                img_gat_attentions = test_analysis["img_gat_attentions"][:, i]
                object_names = test_analysis["object_names"][i]
                object_lens = test_analysis["object_lens"][i]
                rel_trips = test_analysis["rel_trips"][i]
                trip_lens = test_analysis["trip_lens"][i]
                all_attentions = torch.stack([lay[i,:,:,:] for lay in test_analysis["all_attentions"]],dim=0) # 6, 32, 12, 217, 217
                # import pdb;pdb.set_trace()
                # txt_node_filter = test_analysis["txt_node_filter"][:, i]
                # img_node_filter = test_analysis["img_node_filter"][:, i]
                num = batch_idx*args.batch_size+i
                # save input_ids
                # np.save(f"{test_analysis_saved_path}/input_ids/{num}.npy", input_ids.cpu().detach().numpy())
                # gat
                # for gat_idx in range(txt_gat_attentions.shape[0]):
                #     np.save(f"{test_analysis_saved_path}/txt_gat/{num}-layer{gat_idx}.npy", txt_gat_attentions[gat_idx].to(torch.float16).cpu().detach().numpy())
                #     np.save(f"{test_analysis_saved_path}/img_gat/{num}-layer{gat_idx}.npy", img_gat_attentions[gat_idx].to(torch.float16).cpu().detach().numpy())
                # object_names
                # np.save(f"{test_analysis_saved_path}/object_names/{num}.npy", object_names.cpu().detach().numpy())
                # object_lens
                # np.save(f"{test_analysis_saved_path}/object_lens/{num}.npy", object_lens.cpu().detach().numpy())
                # rel_trips
                # np.save(f"{test_analysis_saved_path}/rel_trips/{num}.npy", rel_trips.cpu().detach().numpy())
                # trip_lens
                # np.save(f"{test_analysis_saved_path}/trip_lens/{num}.npy", trip_lens.cpu().detach().numpy())
                # attentions
                # np.save(f"{test_analysis_saved_path}/attentions/{num}.npy", all_attentions.to(torch.float16).cpu().detach().numpy())
                # n_layer, n_head, _, _ = all_attentions.shape
                # for att_layer_idx in range(n_layer):
                #     for att_head_idx in range(n_head):
                #         np.save(f"{test_analysis_saved_path}/attentions/{num}-layer_{att_layer_idx}-head_{att_head_idx}.npy", all_attentions[att_layer_idx, att_head_idx, :, :].to(torch.float16).cpu().detach().numpy())
    
                # print(f"{num}, ok")
                if num == 529: # the sample of cat
                    text = ' '.join(raw_words[i])
                    import pdb;pdb.set_trace()

                cur_src_token = src_tokens[i].cpu().numpy().tolist()
                fw.write(' '.join(raw_words[i])+'\n')
                fw.write('Pred: ')
                for k,v in pred_pairs[i].items():
                    entity_span_ind_list =[]
                    for kk in k:
                        entity_span_ind_list.append(cur_src_token[kk-word_start_index])
                    entity_span = tokenizer.decode(entity_span_ind_list)
                    
                    region_pred, entity_type_ind = v
                    entity_type = ids2label[entity_type_ind[0]]
                    
                    fw.write('('+entity_span+' , '+ str(region_pred)+' , '+entity_type+' ) ')
                fw.write('\n')
                fw.write(' GT : ')
                for k,v in target_pairs[i].items():
                    entity_span_ind_list =[]
                    for kk in k:
                        entity_span_ind_list.append(cur_src_token[kk-word_start_index])
                    entity_span = tokenizer.decode(entity_span_ind_list)
        
                    region_pred, entity_type_ind = v
                    entity_type = ids2label[entity_type_ind[0]]

                    fw.write('('+entity_span+' , '+ str(region_pred)+' , '+entity_type+' ) ')
                fw.write('\n\n')
        res = metric.get_metric()  
        fw.write(str(res))
    return res


# import pdb;pdb.set_trace()
ids2label = {2+i:l for i,l in enumerate(mapping2id.keys())}
model.eval()
test_res = Predict(args,eval_data=test_dataset, model=model, device=device, metric = metric,tokenizer=tokenizer,ids2label=ids2label)
test_f = test_res['f']
print("test: "+str(test_res))