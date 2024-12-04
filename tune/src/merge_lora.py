from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch
import argparse
"""
使用该脚本，将lora的权重合并大base model中
"""
# import sys
# sys.path.append("../")

def merge_lora_to_base_model():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--train_args_file", type=str, default='train_args/pretrain/full/bloom-1b1-pretrain-full.json', help="")
    parser.add_argument("--model_name_or_path", type=str, default='models/Qwen/Qwen-7B-Chat', help="")
    parser.add_argument("--adapter_name_or_path", type=str, help="")
    parser.add_argument("--save_path", type=str,)
    args = parser.parse_args()
    model_name_or_path = args.model_name_or_path
    adapter_name_or_path = args.adapter_name_or_path
    save_path =args.save_path
    # model_name_or_path = 'models/Qwen/Qwen-7B-Chat'
    # adapter_name_or_path = 'output/firefly-qwen-7b-sft-qlora-newexactsqlv1-2'
    # save_path = 'checkpoint/firefly-qwen-7b-qlora-sft-merge-newexactsqlv1-2'

    # model_name_or_path = 'models/meta-llama/llama-2-7b-chat-hf'
    # adapter_name_or_path = 'output/firefly-llama2-7b-sft-qlora-newexactsqlv1-2'
    # save_path = 'checkpoint/firefly-llama2-7b-sft-qlora-newexactsqlv1-2'

    config = AutoConfig.from_pretrained(model_name_or_path,trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(
        adapter_name_or_path,
        trust_remote_code=True,
        # llama不支持fast
        use_fast=False if config.model_type == 'llama' else True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        # device_map='auto',
        device_map={'': 'cpu'}
    )
    model = PeftModel.from_pretrained(model, adapter_name_or_path, device_map={'': 'cpu'})
    model = model.merge_and_unload()

    tokenizer.save_pretrained(save_path)
    model.save_pretrained(save_path)


if __name__ == '__main__':
    merge_lora_to_base_model()
