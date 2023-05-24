#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python test_model.py \
    --val_data ./DATA/OCR-deepsort-IDlen10-num5-len2_punctuation_ASR-add3_20230308/val.json \
    --model_dir ./save_model \
    --output_dir ./result/val

CUDA_VISIBLE_DEVICES=0 python test_model.py \
    --val_data ./DATA/OCR-deepsort-IDlen10-num5-len2_punctuation_ASR-add3_20230308/test.json \
    --model_dir ./save_model \
    --output_dir ./result/test


#gpu=0
# for i in $(seq 12 22)
# do
#     echo $(($i*2000))
#     CUDA_VISIBLE_DEVICES=$(($gpu+$i/2-6)) python test_model.py \
#         --val_data ./DATA/OCR-deepsort-IDlen10-num5-len2_punctuation_ASR-add3_20230308/val.json \
#         --model_dir ./result/OCR-deepsort-IDlen10-num5-len2_punctuation_ASR-add3_20230317_bs12_lr1.5e-05_squad2_ASROCR/checkpoint-$(($i*2000)) \
#         --output_dir ./result_test/OCR-deepsort-IDlen10-num5-len2_punctuation_ASR-add3_20230317_bs12_lr1.5e-05_squad2_ASROCR/checkpoint-$(($i*2000)) &

#     CUDA_VISIBLE_DEVICES=$(($gpu+$i/2-6)) python test_model.py \
#         --val_data ./DATA/OCR-deepsort-IDlen10-num5-len2_punctuation_ASR-add3_20230308/test.json \
#         --model_dir ./result/OCR-deepsort-IDlen10-num5-len2_punctuation_ASR-add3_20230317_bs12_lr1.5e-05_squad2_ASROCR/checkpoint-$(($i*2000)) \
#         --output_dir ./result_test/OCR-deepsort-IDlen10-num5-len2_punctuation_ASR-add3_20230317_bs12_lr1.5e-05_squad2_ASROCR/checkpoint-$(($i*2000))/test &

# done


# for i in $(seq 6 6)
# do
#     echo $(($i*2000))
#     CUDA_VISIBLE_DEVICES=7 python test_model.py \
#         --val_data ./DATA/OCR-deepsort-IDlen10-num5-len2_punctuation_ASR-add3_20230308/val.json \
#         --model_dir ./result/OCR-deepsort-IDlen10-num5-len2_punctuation_ASR-add3_20230314_bs8_lr1e-05_squad2/checkpoint-$(($i*2000)) \
#         --output_dir ./result_test/OCR-deepsort-IDlen10-num5-len2_punctuation_ASR-add3_20230314_bs8_lr1e-05_squad2/checkpoint-$(($i*2000))

#     CUDA_VISIBLE_DEVICES=7 python test_model.py \
#         --val_data ./DATA/OCR-deepsort-IDlen10-num5-len2_punctuation_ASR-add3_20230308/test.json \
#         --model_dir ./result/OCR-deepsort-IDlen10-num5-len2_punctuation_ASR-add3_20230314_bs8_lr1e-05_squad2/checkpoint-$(($i*2000)) \
#         --output_dir ./result_test/OCR-deepsort-IDlen10-num5-len2_punctuation_ASR-add3_20230314_bs8_lr1e-05_squad2/checkpoint-$(($i*2000))/test

# done

#34

# for i in $(seq 26 34)
# do
#     echo $(($i*2000))
#     CUDA_VISIBLE_DEVICES=$(($gpu+$i/2-13)) python test_model.py \
#         --val_data ./DATA/OCR-deepsort-IDlen10-num5-len2_punctuation_ASR-add3_20230308/val.json \
#         --model_dir ./result/OCR-deepsort-IDlen10-num5-len2_punctuation_ASR-add3_20230318_bs8_lr1e-05_squad2/checkpoint-$(($i*2000)) \
#         --output_dir ./result_test/OCR-deepsort-IDlen10-num5-len2_punctuation_ASR-add3_20230318_bs8_lr1e-05_squad2/checkpoint-$(($i*2000)) &

#     CUDA_VISIBLE_DEVICES=$(($gpu+$i/2-13)) python test_model.py \
#         --val_data ./DATA/OCR-deepsort-IDlen10-num5-len2_punctuation_ASR-add3_20230308/test.json \
#         --model_dir ./result/OCR-deepsort-IDlen10-num5-len2_punctuation_ASR-add3_20230318_bs8_lr1e-05_squad2/checkpoint-$(($i*2000)) \
#         --output_dir ./result_test/OCR-deepsort-IDlen10-num5-len2_punctuation_ASR-add3_20230318_bs8_lr1e-05_squad2/checkpoint-$(($i*2000))/test &

# done


#'''test数据集'''
# CUDA_VISIBLE_DEVICES=5 python test_model.py \
#     --val_data ./DATA/OCR-deepsort-IDlen10-num5-len2_punctuation_ASR-add3_20230308/test.json \
#     --model_dir ./result/OCR-deepsort-IDlen10-num5-len2_punctuation_ASR-add3_20230314_bs8_lr1e-05_add_addAsrOcr/checkpoint-22000 \
#     --output_dir ./result_test/OCR-deepsort-IDlen10-num5-len2_punctuation_ASR-add3_20230314_bs8_lr1e-05_add_addAsrOcr/checkpoint-22000/test

# CUDA_VISIBLE_DEVICES=5 python test_model.py \
#     --val_data ./DATA/OCR-deepsort-IDlen10-num5-len2_punctuation_ASR-add3_20230308/test.json \
#     --model_dir ./result/OCR-deepsort-IDlen10-num5-len2_punctuation_ASR-add3_20230314_bs8_lr1e-05_squad2/checkpoint-12000 \
#     --output_dir ./result_test/OCR-deepsort-IDlen10-num5-len2_punctuation_ASR-add3_20230314_bs8_lr1e-05_squad2/checkpoint-12000/test