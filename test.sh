
python test.py \
    --bart_name facebook/bart-base \
    --model_weight ./saved_model/best_model \
    --datapath  data/Twitter10000_v2.0/txt \
    --image_feature_path data/Twitter10000_IETrans \
    --image_annotation_path data/Twitter10000_v2.0/xml \
    --box_num 18 \
    --batch_size 4 \
    --max_len 30 \
    --normalize \

