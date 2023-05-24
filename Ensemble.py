
import json
import numpy as np
from functools import lru_cache
import json
import os
import logging
from collections import Counter
import argparse
import statistics
from statistics import mode
from tqdm import tqdm
import glob

parser = argparse.ArgumentParser()
parser.add_argument('--val_data', required=True, help='path to evaluation dataset')
parser.add_argument('--output_dir', required=True, help="path to result to evaluation")

opt = parser.parse_args()
with open(opt.val_data) as f:
  json_data = json.load(f)

score_path_list = glob.glob('./result_test/*/*/Accuracy_ANLS.json')
score_list = []
for score_path in score_path_list:
    score_file = open(score_path)
    score_f = json.load(score_file)
    score_list.append([score_path,score_f['Accuracy'],score_f['ANLS']])

score_list = sorted(score_list,key=lambda x:float(x[2]),reverse=True)
pred_path_list = [i[0].replace('Accuracy_ANLS.json','predictions_test.json') for i in score_list[:6]]

# pred_path_list =['./result_test/OCR-deepsort-IDlen10-num5-len2_punctuation_ASR-add3_20230314_bs8_lr1e-05_add_addAsrOcr/checkpoint-22000/Accuracy_ANLS.json',\
#     './result_test/OCR-deepsort-IDlen10-num5-len2_punctuation_ASR-add3_20230308_lr2_squad2_seed_666/checkpoint-3000/Accuracy_ANLS.json',\
#     './result_test/OCR-deepsort-IDlen10-num5-len2_punctuation_ASR-add3_20230317_bs12_lr1.5e-05_squad2_ASROCR/checkpoint-8000/Accuracy_ANLS.json',\
#     './result_test/OCR-deepsort-IDlen10-num5-len2_punctuation_ASR-add3_20230308_lr15_squad2_2_seed_666/checkpoint-9000/Accuracy_ANLS.json',\
#     './result_test/OCR-deepsort-IDlen10-num5-len2_punctuation_ASR-add3_20230317_bs8_lr1e-05_squad2/checkpoint-12000/Accuracy_ANLS.json',\
#     './result_test/OCR-deepsort-IDlen10-num5-len2_punctuation_ASR-add3_20230317_bs12_lr1.5e-05_squad2_ASROCR/checkpoint-42000/Accuracy_ANLS.json',\
#     './result_test/OCR-deepsort-IDlen10-num5-len2_punctuation_ASR-add3_20230309_bs8_lr1e-05/checkpoint-12000/Accuracy_ANLS.json',\
#     './result_test/OCR-deepsort-IDlen10-num5-len2_punctuation_ASR-add3_20230317_bs18_lr2e-05_squad2/checkpoint-8000/Accuracy_ANLS.json',\
#     './result_test/OCR-deepsort-IDlen10-num5-len2_punctuation_ASR-add3_20230317_bs8_lr1e-05_squad2_ASROCR/checkpoint-16000/Accuracy_ANLS.json',\
#     './result_test/OCR-deepsort-IDlen10-num5-len2_punctuation_ASR-add3_20230317_bs12_lr1.5e-05_squad2/checkpoint-18000/Accuracy_ANLS.json',\
#     './result_test/OCR-deepsort-IDlen10-num5-len2_punctuation_ASR-add3_20230318_bs8_lr1e-05_squad2/checkpoint-54000/Accuracy_ANLS.json']
# pred_path_list = [i.replace('Accuracy_ANLS.json','predictions_test.json') for i in pred_path_list[:6]]  
'''用test做结果保存'''
pred_path_list = [i[0].replace('Accuracy_ANLS.json','test/predictions_test.json') for i in score_list[:6]]

print('文件数量:',len(pred_path_list))
print('最好指标:')
for i in score_list[:6]:
    print(i[0])
    print(i[1:])

result_list = {}
for pred_path in pred_path_list:
    prediction_file = open(pred_path)
    pred_f = json.load(prediction_file)
    for key,value in pred_f.items():
        if key not in result_list.keys():
            result_list[key] = [value]
        else:
            result_list[key].append(value)

pred_f = {}
for key,value in result_list.items():
    if Counter(value).most_common(1)[0][1] == 1:
        pred_f[key] = value[0]
    else:
        pred_f[key] = Counter(value).most_common(1)[0][0]
    #print(Counter(value).most_common(1)[0])


'''单独文件测试'''
#pred_f = json.load(open('./result_submission/predictions_val.json'))

with open(os.path.join(opt.output_dir, "predictions_test.json"), 'w') as f:
  json.dump(pred_f, f)

res = []
for q_id, pred_answer in pred_f.items():
	res.append({
	"answer": pred_answer,
	"questionId": q_id
	})
with open(os.path.join(opt.output_dir, "submission_test.json"), 'w') as f:
  json.dump(res, f)

exit()
def lev_dist(a, b):
    
	@lru_cache(None)  # for memorization
	def min_dist(s1, s2):

		if s1 == len(a) or s2 == len(b):

			return len(a) - s1 + len(b) - s2
		if a[s1] == b[s2]:
			return min_dist(s1 + 1, s2 + 1)

		return 1 + min(
      min_dist(s1, s2 + 1),      # insert character
      min_dist(s1 + 1, s2),      # delete character
      min_dist(s1 + 1, s2 + 1),  # replace character
    )

	return min_dist(0, 0)

acc = 0
ls = 0

all_ori_ids = []
all_gt_ans = []
for j in json_data:
	all_ori_ids.append(j["qas"][0]["id"])
	all_gt_ans.append(j["qas"][0]["answers"][0]["text"])

for i in tqdm(range(len(all_ori_ids))):
	all_ans = []
	gt_ans = all_gt_ans[i]
	for l in json_data:
		if str(all_ori_ids[i]) == str(l["qas"][0]["id"]):
			for k, v in pred_f.items():
				if str(l["qas"][0]["id"]) == str(k):
					all_ans.append(v)


	# print(all_ans, gt_ans)
	c = Counter(all_ans)
	voted_ans_l=(c.most_common(1))
	voted_ans = (voted_ans_l[0][0])

	if str(gt_ans.lower()) == str(voted_ans).lower():
		acc+=1

	ss = lev_dist(str(gt_ans).lower(), str(voted_ans).lower())
    
    
	ss=1-(ss/max(len(gt_ans.lower()), len(str(voted_ans).lower())))

	if ss>=0.5:
		ss=ss
	else:
		ss=0
	ls+=ss

print("Accuracy is: ", ((acc/(len(json_data))*100)))
print("ANLS: ", ls/max(len(json_data), len(pred_f))*100)


'''保存结果'''

# result = {}
# result['Accuracy'] = acc/(len(json_data))*100
# result['ANLS'] = ls/max(len(json_data), len(pred_f))*100
# with open(opt.output_dir + "/Accuracy_ANLS.json",'w') as f:
# 	json.dump(result,f)



