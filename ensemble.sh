#!/bin/bash

python Ensemble.py \
    --val_data ./DATA/OCR-deepsort-IDlen10-num5-len2_punctuation_ASR-add3_20230308/val.json \
    --output_dir ./result_submission
