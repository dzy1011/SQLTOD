import random
import json
import jsonlines
import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str)
parser.add_argument("--out_file", type=str)
args = parser.parse_args()
data_path = args.data_path
out_file = args.out_file
# dataset_version = "sql3-2-dial-80-linear"
# data_path = "dialogue/train_data_mixed_linear/{}.jsonl".format(dataset_version)
all_data = []
with open(data_path,"r") as f:
    for line in f.readlines():
        line = line.strip()
        line_dic = json.loads(line)
        all_data.append(line_dic)
dataset_len = len(all_data)
# print(dataset_len)
data_id = [i for i in range(0, dataset_len)]
random.shuffle(data_id)
# print(len(data_id))
# print(data_id[:5])
os.makedirs(os.path.dirname(out_file),exist_ok=True)
# out_file = "dialogue/train_data_mixed_linear/shuffled/{}.jsonl".format(dataset_version)
with jsonlines.open(out_file,"w") as f:
    for id in data_id:
        f.write(all_data[id])