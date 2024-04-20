for se in   '42' '2022' '16' '34' '2023' '25' '23'
do
python train.py \
    --bart_name facebook/bart-base \
    --n_epochs 30 \
    --seed $se \
    --datapath   ../GMNER_data/Twitter10000_v2.0/txt \
    --image_feature_path ../GMNER_data/sgcls_thrs02_features_rel01_v1 \
    --image_annotation_path  ../GMNER_data/Twitter10000_v2.0/xml \
    --lr 3e-5 \
    --box_num 18 \
    --batch_size 32 \
    --max_len 30 \
    --save_model 1 \
    --normalize \
    --use_kl \
    --radar_lr_rate 50 \
    --gnn_drop 0.5
done
#  '42' '2022' '16' '34' '2023' '25' '23'
# '42' '2022' '16' '34' '2023' '25' '23'
# '42' '16' '34' '2023' '25' '23' '2022'
