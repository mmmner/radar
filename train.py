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

fitlog.debug()
fitlog.set_log_dir('logs')
load_dataset_seed = 100
fitlog.add_hyper(load_dataset_seed,'load_dataset_seed')




import argparse
fitlog.set_rng_seed(load_dataset_seed)

parser = argparse.ArgumentParser()

parser.add_argument('--bart_name', default='facebook/bart-large', type=str)
parser.add_argument('--datapath', default='./Twitter_GMNER/txt/', type=str)
parser.add_argument('--image_feature_path',default='./data/Twitter_GMNER_vinvl', type=str)
parser.add_argument('--image_annotation_path',default='./Twitter_GMNER/xml/', type=str)
parser.add_argument('--region_loss_ratio',default='1.0', type=float)
parser.add_argument('--box_num',default='16', type=int)
parser.add_argument('--normalize',default=False, action = "store_true")
parser.add_argument('--use_kl',default=False,action ="store_true")
parser.add_argument('--lr', default=1e-5, type=float)
parser.add_argument('--n_epochs', default=30, type=int)
parser.add_argument('--max_len', default=30, type=int)
parser.add_argument('--batch_size',default=16,type=int)
parser.add_argument('--seed',default=42,type=int)
parser.add_argument("--save_model",default=1,type=int)
parser.add_argument("--save_path",default='save_models/best',type=str)
parser.add_argument("--radar_lr_rate",default=50,type=float)
parser.add_argument("--gnn_drop",default=0.5,type=float)
parser.add_argument("--ib_weight",default=0.01,type=float)
parser.add_argument("--num_layers",default=2,type=int)

# parser.add_argument("--log",default='./logs',type=str)
parser.add_argument('--device', default=None)#'cuda:0'
args= parser.parse_args()
import sys
entry_path = sys.argv[0]
print(entry_path)
print(args)
# import pdb;pdb.set_trace()
now_time=datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
output_dir='logs/'
if not os.path.exists(output_dir):  # 判断是否存在文件夹如果不存在则创建文件夹
    os.makedirs(output_dir)
# if save_model
save_dir='saved_model/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
args.save_path=os.path.join('./saved_model/',now_time+ "_best_model")

# output records
output_file = os.path.join(output_dir, now_time+"_records.txt")
writer = open(output_file, "a")
writer.write(str(args)+'\n')
writer.write('seed: '+str(args.seed)+'\n')
writer.flush()

dataset_name = 'twitter-ner'
args.length_penalty = 1

device = args.device

args.target_type = 'word'
args.schedule = 'linear'
args.decoder_type = 'avg_feature'
args.num_beams = 1   
args.use_encoder_mlp = 1
args.warmup_ratio = 0.01
eval_start_epoch = 0


if 'twitter' in dataset_name:  
    max_len, max_len_a = args.max_len, 0.6
else:
    print("Error dataset_name!")


if isinstance(args.decoder_type, str) and args.decoder_type.lower() == 'none':
    args.decoder_type = None
demo = False 
region_dim=2048
refresh_data=False

if '15' in args.datapath:
    cache_name='cache/twitter15'+str(args.box_num)
    dataset_saving='twitter15'
elif '17' in args.datapath:
    cache_name='cache/twitter17'+str(args.box_num)
    dataset_saving='twitter17'
else:
    cache_name='cache/gmner'+str(args.box_num)
    dataset_saving='gmner'
if demo:
    cache_name+='Demo'
@cache_results(_cache_fp=cache_name, _refresh=refresh_data)
def get_data():

    pipe = BartNERPipe(image_feature_path=args.image_feature_path, 
                       image_annotation_path=args.image_annotation_path,
                       max_bbox =args.box_num,
                       normalize=args.normalize,
                       tokenizer=args.bart_name, 
                       target_type=args.target_type,region_dim=region_dim)
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

# torch.manual_seed(args.seed)
fitlog.set_rng_seed(args.seed)
# import pdb;pdb.set_trace()
model = BartSeq2SeqModel.build_model(args.bart_name, tokenizer, label_ids=label_ids, decoder_type=args.decoder_type,
                                     use_encoder_mlp=args.use_encoder_mlp,box_num = args.box_num,gnn_drop=args.gnn_drop,args=args)


vocab_size = len(tokenizer)

model = SequenceGeneratorModel(model, bos_token_id=bos_token_id,
                               eos_token_id=eos_token_id, 
                               max_length=max_len, max_len_a=max_len_a,num_beams=args.num_beams, do_sample=False,
                               repetition_penalty=1, length_penalty=args.length_penalty, pad_token_id=eos_token_id,
                               restricter=None, top_k = 1
                               )

## parameter scale
total = sum([param.nelement() for param in model.parameters()])
print("Number of parameter: %.2fM" % (total/1e6))
##

import torch
if device is None:
    if torch.cuda.is_available():
            device = 'cuda'
    else:
        device = 'cpu'
print('device: '+str(device) + '\n')
writer.write(str(device) + '\n')
writer.flush()

checkpoint=None
if checkpoint:
    model.load_state_dict(torch.load(checkpoint,map_location=device))
    print('checkpoint loaded:',checkpoint)


radar_params=[]
gmner_params=[]

for name, param in model.named_parameters(): #275
    if 'radar' in name:
        radar_params.append(param) #269
    else:
        gmner_params.append(param)
params_=[{'params': radar_params, 'lr': args.lr * args.radar_lr_rate},
                      {'params':gmner_params,'lr':args.lr}]
optimizer = optim.AdamW(params_)




metric = Seq2SeqSpanMetric(eos_token_id, num_labels=len(label_ids), region_num =args.box_num, target_type=args.target_type,print_mode = False )

train_dataset = data_bundle.get_dataset('train')
eval_dataset = data_bundle.get_dataset('dev')
test_dataset = data_bundle.get_dataset('test')
print(train_dataset[:3])

device = torch.device(device)
model.to(device)

def Training(args, train_idx, train_data, model, device, optimizer):
    
    train_sampler = BucketSampler(seq_len_field_name='src_seq_len',batch_size=args.batch_size)   # 带Bucket的 Random Sampler. 可以随机地取出长度相似的元素
    train_data_iterator = DataSetIter(train_data, batch_size=args.batch_size, sampler=train_sampler)
    
    train_loss = 0.
    train_region_loss = 0.
    idx=0
    # import pdb;pdb.set_trace()

    '''
    for name,param in model.named_parameters():
        if name in not_gcn_params and param.requires_grad:
            param.requires_grad = False
    '''
    for batch_x, batch_y in (train_data_iterator):
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

        rel_adjs=batch_x['relations']
        # rel text related:
        rel_bpes=batch_x['rel_bpes']
        rel_lens=batch_x['rel_lens']
        rel_trips=batch_x['rel_trips']
        trip_lens=batch_x['trip_lens']
        # import pdb;pdb.set_trace()
        # print(idx)
        # if idx==57: #6999: # 218
        #     import pdb;pdb.set_trace()
        results = model(src_tokens,image_feature, tgt_tokens, src_seq_len=src_seq_len, tgt_seq_len=tgt_seq_len, first=first,object_names=object_names,obj_lens=obj_lens,rel_adjs=rel_adjs,rel_lens=rel_lens,rel_trips=rel_trips,trip_lens=trip_lens)
        pred, region_pred = results['pred'],results['region_pred'],   ## logits:(bsz,tgt_len,class+max_len)  region_logits:(??,8)

        loss, region_loss = get_loss(tgt_tokens, tgt_seq_len, pred, region_pred,region_label,use_kl=args.use_kl)

        train_loss += loss.item()
        train_region_loss += region_loss.item()

        all_loss = loss + args.region_loss_ratio * region_loss 

        all_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        idx+=1
    print("train_loss: %f"%(train_loss))
    print("train_region_loss: %f"%(train_region_loss))
    return train_loss, train_region_loss

def Inference(args,eval_data, model, device, metric):
    data_iterator = DataSetIter(eval_data, batch_size=args.batch_size * 2, sampler=SequentialSampler())
    # for batch_x, batch_y in tqdm(data_iterator, total=len(data_iterator)):
    for batch_x, batch_y in (data_iterator):
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
        # import pdb;pdb.set_trace()//
        results = model.predict(src_tokens,image_feature, src_seq_len=src_seq_len, first=first,object_names=object_names,obj_lens=obj_lens,rel_adjs=rel_adjs,rel_lens=rel_lens,rel_trips=rel_trips,trip_lens=trip_lens)
        
        pred,region_pred = results['pred'],results['region_pred']   ## logits:(bsz,tgt_len,class+max_len)  region_logits:(??,8)
        
        metric.evaluate(target_span, pred, tgt_tokens, region_pred,region_label,cover_flag)
    res = metric.get_metric()  ## {'f': 20.0, 'rec': 16.39, 'pre': 25.64, 'em': 0.125, 'uc': 0}
    return res



max_dev_f = 0.
max_test_by_dev = 0.
max_test_f = 0.
best_dev = {}
best_test = {}
best_dev_corresponding_test = {}

for train_idx in range(args.n_epochs):
    print("-"*12+"Epoch: "+str(train_idx)+"-"*12)

    model.train()
    train_loss, train_region_loss = Training(args,train_idx=train_idx,train_data=train_dataset, model=model, device=device,
                                                optimizer=optimizer)
    

    model.eval()
    
    for i in range(6):
        # model.seq2seq_model.encoder.bart_encoder.layers[0].self_attn.radar_obj2img_w in 
        if hasattr(model.seq2seq_model.encoder.bart_encoder.layers[i].self_attn, 'obj2img_w'):
            print('obj2img_w of layer_'+str(i)+':',model.seq2seq_model.encoder.bart_encoder.layers[i].self_attn.obj2img_w.item())
        if hasattr(model.seq2seq_model.encoder.bart_encoder.layers[i].self_attn, 'img2obj_w'):
            print('img2obj_w of layer_'+str(i)+':',model.seq2seq_model.encoder.bart_encoder.layers[i].self_attn.img2obj_w.item())
    
    dev_res = Inference(args,eval_data=eval_dataset, model=model, device=device, metric = metric)
    dev_f = dev_res['f']
    print("dev: "+str(dev_res))

   
    

    # train_res = Inference(args,eval_data=train_dataset, model=model, device=device, metric = metric)
    # train_f = train_res['f']
    # print("train: "+str(train_res))



    if dev_f >= max_dev_f:
        test_res = Inference(args,eval_data=test_dataset, model=model, device=device, metric = metric)
        test_f = test_res['f']
        print("test: "+str(test_res))
        max_dev_f = dev_f 
        max_test_by_dev=test_f
        if args.save_model:
            model_to_save = model.module if hasattr(model, 'module') else model  
            torch.save(model_to_save.state_dict(), args.save_path)
        best_dev = dev_res
        best_dev['epoch'] = train_idx
        best_dev_corresponding_test = test_res
        print('best_test:'+str(test_res))
        best_dev_corresponding_test['epoch'] = train_idx
        
   
    # if test_f >= max_test_f:
    #     max_test_f = test_f 
    #     best_test = test_res
    #     best_test['epoch'] = train_idx
import os
if args.save_model:
    os.system('mv '+args.save_path+' '+args.save_path+'_'+str(max_test_by_dev)+dataset_saving)
print("best_dev: "+str(best_dev))
print("best_dev_corresponding_test: "+str(best_dev_corresponding_test))
print("GMNER MNER EEG")
print(best_dev_corresponding_test['f'],best_dev_corresponding_test['mner_f'],best_dev_corresponding_test['eeg_f'],)
# import pdb;pdb.set_trace()
writer.write("best_dev_corresponding_test: "+str(best_dev_corresponding_test)+'\n')
writer.write("GMNER MNER EEG\n"+str(best_dev_corresponding_test['f'])+' '+str(best_dev_corresponding_test['mner_f'])+' '+str(best_dev_corresponding_test['eeg_f'])+'\n')
# import pdb;pdb.set_trace()
# if args.save_path and args.save_model:
#     print("-"*12+'Predict'+'-'*12)
#     ids2label = {2+i:l for i,l in enumerate(mapping2id.keys())}

#     model_path = args.save_path.rsplit('/')
#     args.pred_output_file = '/'.join(model_path[:-1])+'/pred_'+model_path[-1]+'.txt'

#     model.load_state_dict(torch.load(args.save_path))
#     model.to(device)

#     print(test_dataset[:3])
#     test_dataset.set_target('raw_words', 'raw_target')


#     model.eval()
#     test_res = Predict(args,eval_data=test_dataset, model=model, device=device, metric = metric,tokenizer=tokenizer,ids2label=ids2label)
#     test_f = test_res['f']
#     print("test: "+str(test_res))
