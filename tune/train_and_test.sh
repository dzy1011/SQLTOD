
cd "$(dirname "$0")"

if [ $? -eq 0 ]; then
  CUDA_VISIBLE_DEVICES=0 python src/train.py\
  --train_args_file src/train_args/sft/qlora/qwen-7b-sft-qlora_final.json
fi

if [ $? -eq 0 ]; then
  CUDA_VISIBLE_DEVICES=0 python src/merge_lora.py\
  --model_name_or_path src/models/Qwen/Qwen-7B-Chat\
  --adapter_name_or_path src/output/firefly-qwen-7b-sft-qlora-sql3-2-dial-50-3epo-linear\
  --save_path src/checkpoint/firefly-qwen-7b-qlora-sft-merge-sql3-2-dial-50-3epo-linear
fi


if [ $? -eq 0 ]; then
    CUDA_VISIBLE_DEVICES=0 python src/sql/model_generate_sql.py\
    --model_name_or_path src/checkpoint/firefly-qwen-7b-qlora-sft-merge-sql3-2-dial-50-3epo-linear\
    --template_name qwen\
    --trainsql_version sql3-2-dial-50-3epo-linear
fi


if [ $? -eq 0 ]; then
  CUDA_VISIBLE_DEVICES=0 python src/dialogue/generate_data/sql_extract_kb.py\
  --kb_version sql3-2-dial-50-3epo-linear\
  --model_version sql3-2-dial-50-3epo-linear
fi


if [ $? -eq 0 ]; then
  CUDA_VISIBLE_DEVICES=0 python src/dialogue/generate_data/prepare_conversation.py\
  --kb_version sql3-2-dial-50-3epo-linear
fi


if [ $? -eq 0 ]; then
  CUDA_VISIBLE_DEVICES=0 python src/dialogue/test/test_dialogue.py\
  --model_name_or_path src/checkpoint/firefly-qwen-7b-qlora-sft-merge-sql3-2-dial-50-3epo-linear\
  --dataset MultiWOZ\
  --template_name qwen\
  --model_version sql3-2-dial-50-3epo-linear\
  --kb_version sql3-2-dial-50-3epo-linear\
  --kb_type small
fi

if [ $? -eq 0 ]; then
  CUDA_VISIBLE_DEVICES=0 python src/dialogue/test/test_dialogue.py\
  --model_name_or_path src/checkpoint/firefly-qwen-7b-qlora-sft-merge-sql3-2-dial-50-3epo-linear\
  --dataset CamRest\
  --template_name qwen\
  --model_version sql3-2-dial-50-3epo-linear\
  --kb_type small\
  --kb_version sql3-2-dial-50-3epo-linear
fi


if [ $? -eq 0 ]; then
  CUDA_VISIBLE_DEVICES=0 python src/dialogue/test/test_dialogue.py\
  --model_name_or_path src/checkpoint/firefly-qwen-7b-qlora-sft-merge-sql3-2-dial-50-3epo-linear\
  --dataset SMD\
  --template_name qwen\
  --model_version sql3-2-dial-50-3epo-linear\
  --kb_type small\
  --kb_version sql3-2-dial-50-3epo-linear
fi