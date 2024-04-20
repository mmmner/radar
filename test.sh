
python test.py \
    --bart_name facebook/bart-base \
    --model_weight ./saved_model/best_model \
    --datapath  ../GMNER_data/Twitter10000_v2.0/txts/total_txt \
    --image_feature_path ../GMNER_data/sgcls_thrs02_features_rel01_v1 \
    --image_annotation_path ../GMNER_data/Twitter10000_v2.0/xml \
    --box_num 18 \
    --batch_size 4 \
    --max_len 30 \
    --normalize \

# ./Twitter10000/txt
