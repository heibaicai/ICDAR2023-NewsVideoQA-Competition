from simpletransformers.question_answering import QuestionAnsweringModel
import json
import os
import logging
import argparse
import random

parser = argparse.ArgumentParser()
parser.add_argument('--val_data', required=True, help='path to evaluation dataset')
parser.add_argument('--train_data_v1', required=True, help='path to train dataset')
parser.add_argument('--train_data_v2', required=True, help='path to train dataset')
parser.add_argument('--best_model_dir', required=True, help="path to saved_model to evaluation")
parser.add_argument('--output_dir', required=True, help="path to result to evaluation")
parser.add_argument('--pre_model_path', required=True, help="path to pre_model")
parser.add_argument('--manualSeed', type=int, default=100, help='for random seed setting')

opt = parser.parse_args()


def data_aug(data):
  new_data = []
  for i in data:
    answers = i['qas'][0]['answers'][0]['text']
    context = i['context']
    answers_list = answers.split(' ')
    if len(answers_list) <= 2:
      answers_new = ''.join(answers_list)
    else:
      num = random.choice(range(len(answers_list)))
      answers_new = ' '.join(answers_list[:num]) + ' '.join(answers_list[num:])

    
    i['qas'][0]['answers'][0]['text'] = answers_new
    i['context'] = context.replace(answers,answers_new)
    new_data.append(i)
  return new_data

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

# training parameters; given below is the default setting used for bert large models
#for BERT large you need at least 4 GPUS with 11 GB of GPU momory
# for bert base models one GPU is sufficient for a batch size of 32
# make lr higher if you train with larger batch size

model_args = {"train_batch_size": 8,
               "n_gpu":2, 
               "eval_batch_size": 4, 
               'max_answer_length': 10,  
               'num_train_epochs': 40, 
               'output_dir': opt.output_dir, 
               'best_model_dir': opt.best_model_dir, 
               'evaluate_during_training': False, 
               'fp16': True, 
               'use_cached_eval_features':False, ##如果为True,不同的验证集会出bug
               'save_eval_checkpoints': False, 
               'save_model_every_epoch': False, 
               'save_steps': 2000,
               'max_seq_length': 512, 
               'doc_stride': 128, 
               'do_lower_case': True, 
               'gradient_accumulation_steps': 1, 
               'learning_rate': 1e-05,
               'manual_seed': opt.manualSeed}


# if you want to fine tune a model locally saved or say you want to continue training a model previously saved give location of the dir where the model is
# model = QuestionAnsweringModel('bert', '/ssd_scratch/cvit/soumyajahagirdar/BERT/experiments/ocr_cc/checkpoint-49920-epoch-8/', args=model_args)


# if you want to fine tune a pretrained model from pytorch trasnformers model zoo (https://huggingface.co/transformers/pretrained_models.html), you can directly give the model name ..the pretrained model will be downloadef first to a cache dir 
# here the model we are fine tuning is bert-large-cased-whole-word-masking-finetuned-squad
#model = QuestionAnsweringModel('bert', 'bert-large-cased-whole-word-masking-finetuned-squad', args=model_args, use_cuda=True)

model = QuestionAnsweringModel('bert',opt.pre_model_path, args=model_args, use_cuda=True)

with open(opt.train_data_v1) as f:
  train_data_v1 = json.load(f)

with open(opt.train_data_v2) as f:
  train_data_v2 = json.load(f) 

with open('./DATA/ASR_add3_OCR-deepsort-IDlen10-num5-len2_punctuation_20230314/train_v1.json') as f:
  train_data_v3 = json.load(f)

with open('./DATA/ASR_add3_OCR-deepsort-IDlen10-num5-len2_punctuation_20230314/train_v2.json') as f:
  train_data_v4 = json.load(f)

# train_data_v3 = data_aug(train_data_v1)
# train_data_v4 = data_aug(train_data_v2)

train_data = train_data_v1 + train_data_v2
train_data = train_data + train_data_v3 + train_data_v4


with open(opt.val_data) as f:
  dev_data = json.load(f)
#import ipdb;ipdb.set_trace()
model.train_model(train_data, show_running_loss= True, eval_data=dev_data)
                           
