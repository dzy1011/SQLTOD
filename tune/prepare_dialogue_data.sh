set -eux

# python SQLTOD/tune/src/dialogue/prepare_random_linear.py\
#   --inp_file src/origindata/CamRest/train.json\
#   --out_file dialogue/train_data-80-linear/CamRest_train.jsonl

#!/bin/bash


cd "$(dirname "$0")"

## ! create the part of sql dataset
python src/sql/prepare_train.py

## ! create the part of dialogue dataset
datasets=("CamRest" "MultiWOZ" "SMD")  
noise_percent=50 # noise percentage in dataset

for dataset in "${datasets[@]}"; do

  python src/dialogue/prepare_random_linear.py \
    --inp_file src/origindata/${dataset}/train.json \
    --out_file src/dialogue/train_data-${noise_percent}-linear/${dataset}_train.jsonl
done

# mkdir src/dialogue/train_data-${noise_percent}-linear
##! merge dialogue dataset and sql dataset
python src/dialogue/merge_file.py\
  src/dialogue/train_data-${noise_percent}-linear/CamRest_train.jsonl\
  src/dialogue/train_data-${noise_percent}-linear/MultiWOZ_train.jsonl\
  src/dialogue/train_data-${noise_percent}-linear/SMD_train.jsonl\
  src/sql/sftfinal_sql/all_train.jsonl\
  --output_file src/dialogue/train_data-${noise_percent}-linear/sql_final_dial_${noise_percent}.jsonl


##! shuffle
python src/dialogue/shuffle_data.py\
  --data_path src/dialogue/train_data-${noise_percent}-linear/sql_final_dial_${noise_percent}.jsonl\
  --out_file  src/dialogue/train_data-${noise_percent}-linear/sql_final_dial_${noise_percent}_shuffled.jsonl
