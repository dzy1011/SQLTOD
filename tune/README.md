
## Directory
+ `final_sql` the final sql-annotated datasets
+ `src` 
    + `checkpoint` the merged model weights
    + `dialogue` part for dialogue
        + `evaluate` evaluation code for dialogue outputs
        + `generate_data` code for generate dialogue training dataset and knowledge base files
        + `test` model outputs 
    + `sql` part for sql
        + `sftfinal_sql` the conversation format of the final sql-annotated datasets
        + `test` model outputs
    + `output` model trained by QLora
    + `models` Qwen and llama2
    + `origindata` original datasets
    + `train_args` Qlora config json files
## Usage
### Preparation
Prepare datasets for training and testing.

> bash ./prepare_dialogue_data.sh

It converts the `final_sql` data and dialogue data into conversation format and merges them into a new file.
### Train and test

Use the prepared data to instruct-finetune the models, then test the finetuend models on sql dataset and dialogue dataset.

> bash ./train_and_test.sh

### evaluate the quality of generated sql
Caculate the EX and Relax-EX of the generated SQL.
Take MultiWOZ dataset as an example.
```shell
python src/sql/evaluate.py\
    --json_path src/sql/test/qwen_output_sql3-2-dial-50-3epo-linear/MultiWOZ_test.json
```    

### evaluate the quality of generated dialogue
Take MultiWOZ dataset as an example.
Move the test output of the model into te direcotory `src/dialogue/evalute`

```shell 
python src/dialogue/evaluate/evaluate.py\ 
    --dataset MultiWOZ\
    --pred_file RES_SMD-sql3-2-dial-50-3epo-linear.jsonl
```
