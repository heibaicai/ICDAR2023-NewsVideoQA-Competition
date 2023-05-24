#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1 python finetune_pretrained_model.py \
    --val_data ./DATA/OCR-deepsort-IDlen10-num5-len2_punctuation_ASR-add3_20230308/val.json \
    --train_data_v1 ./DATA/OCR-deepsort-IDlen10-num5-len2_punctuation_ASR-add3_20230308/train_v1.json \
    --train_data_v2 ./DATA/OCR-deepsort-IDlen10-num5-len2_punctuation_ASR-add3_20230308/train_v2.json \
    --best_model_dir ./save_model/OCR-deepsort-IDlen10-num5-len2_punctuation_ASR-add3_20230318_bs8_lr1e-05_squad2 \
    --output_dir ./result/OCR-deepsort-IDlen10-num5-len2_punctuation_ASR-add3_20230318_bs8_lr1e-05_squad2 \
    --pre_model_path /users/caizhuang/ICDAR2023/NewsVideoQA/baselines/BERT/pre_model/deepset-bert-large-uncased-whole-word-masking-squad2 \
    --manualSeed 100 