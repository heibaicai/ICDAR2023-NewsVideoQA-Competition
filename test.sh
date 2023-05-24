#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python test_model.py \
    --val_data ./DATA/OCR-deepsort-IDlen10-num5-len2_punctuation_ASR-add3_20230308/val.json \
    --model_dir ./save_model \
    --output_dir ./result/val

CUDA_VISIBLE_DEVICES=0 python test_model.py \
    --val_data ./DATA/OCR-deepsort-IDlen10-num5-len2_punctuation_ASR-add3_20230308/test.json \
    --model_dir ./save_model \
    --output_dir ./result/test
    