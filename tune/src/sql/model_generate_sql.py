import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from transformers import AutoTokenizer, AutoConfig, AddedToken
import torch
from loguru import logger
import copy
import json
import sqlite3
from tqdm import tqdm
import argparse
# import sys
# sys.path.append("../../")
from component.utils import ModelUtils
from component.template import template_dict


def build_prompt_chatglm3(tokenizer, query, history, system=None):
    history.append({"role": 'user', 'message': query})
    # system
    input_ids = tokenizer.get_prefix_tokens() + \
                [tokenizer.get_command(f"<|system|>")] + \
                tokenizer.encode(system, add_special_tokens=False)
    # convs
    for item in history:
        role, message = item['role'], item['message']
        if role == 'user':
            tokens = [tokenizer.get_command(f"<|user|>")] + \
                     tokenizer.encode(message, add_special_tokens=False) + \
                     [tokenizer.get_command(f"<|assistant|>")]
        else:
            tokens = tokenizer.encode(message, add_special_tokens=False) + [tokenizer.eos_token_id]
        input_ids += tokens

    return input_ids


def build_prompt(tokenizer, template, query, history, system=None):
    template_name = template.template_name
    system_format = template.system_format
    user_format = template.user_format
    assistant_format = template.assistant_format
    system = system if system is not None else template.system

    if template_name == 'chatglm2':
        prompt = tokenizer.build_prompt(query, history)
        input_ids = tokenizer.encode(prompt)
    elif template_name == 'chatglm3':
        input_ids = build_prompt_chatglm3(tokenizer, query, history, system)
    else:
        history.append({"role": 'user', 'message': query})
        input_ids = []

        # setting system information
        if system_format is not None:
            # system信息不为空
            if system is not None:
                system_text = system_format.format(content=system)
                input_ids = tokenizer.encode(system_text, add_special_tokens=False)
        # concat conversation
        for item in history:
            role, message = item['role'], item['message']
            if role == 'user':
                message = user_format.format(content=message, stop_token=tokenizer.eos_token)
            else:
                message = assistant_format.format(content=message, stop_token=tokenizer.eos_token)
            tokens = tokenizer.encode(message, add_special_tokens=False)
            input_ids += tokens
    input_ids = torch.tensor([input_ids], dtype=torch.long)

    return input_ids


def load_tokenizer(model_name_or_path):
    # config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
    # 加载tokenzier
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        use_fast=False
        # llama不支持fast
        # use_fast=False if config.model_type == 'llama' else True
    )

    if tokenizer.__class__.__name__ == 'QWenTokenizer':
        tokenizer.pad_token_id = tokenizer.eod_id
        tokenizer.bos_token_id = tokenizer.eod_id
        tokenizer.eos_token_id = tokenizer.eod_id
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # assert tokenizer.pad_token_id is not None, "pad_token_id should not be None"
    return tokenizer

def remove_duplicates_by_id(items):
    unique_items = []
    seen_ids = set()
    for item in items:
        if item['id'] not in seen_ids:
            unique_items.append(item)
            seen_ids.add(item['id'])
    return unique_items

def main():
    # 使用合并后的模型进行推理
    # model = 'qwen'
    parser = argparse.ArgumentParser()
    # parser.add_argument("--train_args_file", type=str, default='train_args/pretrain/full/bloom-1b1-pretrain-full.json', help="")
    parser.add_argument("--model_name_or_path", type=str, default='checkpoint/firefly-qwen-7b-qlora-sft-merge-exactsqlv1', help="")
    parser.add_argument("--template_name", type=str, default="qwen", help="")
    parser.add_argument("--trainsql_version", type=str, default="newexactsqlv1-2")
    args = parser.parse_args()
    template_name = args.template_name
    model_name_or_path = args.model_name_or_path
    trainsql_version= args.trainsql_version
    datasets = ["CamRest","MultiWOZ","SMD"]
    modes = ["test"]
    # trainsql_version = "newexactsqlv1-2"


    # model_name_or_path = 'checkpoint/firefly-qwen-7b-qlora-sft-merge-exactsqlv1'
    # template_name = 'qwen'
    # model_name_or_path = 'checkpoint/firefly-llama2-7b-sft-qlora-newexactsqlv1-2'
    # template_name = 'llama2'
    adapter_name_or_path = None

    # model_name_or_path = '01-ai/Yi-6B-Chat'
    # template_name = 'yi'
    # adapter_name_or_path = None

    template = template_dict[template_name]
    # 是否使用4bit进行推理，能够节省很多显存，但效果可能会有一定的下降
    load_in_4bit = False
    # 生成超参配置
    max_new_tokens = 500
    top_p = 0.9
    temperature = 0.35
    repetition_penalty = 1.0

    # 加载模型
    logger.info(f'Loading model from: {model_name_or_path}')
    logger.info(f'adapter_name_or_path: {adapter_name_or_path}')
    model = ModelUtils.load_model(
        model_name_or_path,
        load_in_4bit=load_in_4bit,
        adapter_name_or_path=adapter_name_or_path
    ).eval()
    tokenizer = load_tokenizer(model_name_or_path if adapter_name_or_path is None else adapter_name_or_path)
    if template_name == 'chatglm2':
        stop_token_id = tokenizer.eos_token_id
    elif template_name == 'chatglm3':
        stop_token_id = [tokenizer.eos_token_id, tokenizer.get_command("<|user|>"), tokenizer.get_command("<|observation|>")]
    else:
        if template.stop_word is None:
            template.stop_word = tokenizer.eos_token
        stop_token_id = tokenizer.encode(template.stop_word, add_special_tokens=False)
        assert len(stop_token_id) == 1
        stop_token_id = stop_token_id[0]

    history = []
    intent2table = {
        "restaurant":"RESTAURANT",
        "schedule":"SCHEDULE",
        "attraction":"ATTRACTION",
        "hotel":"HOTEL",
        "weather":"WEATHER",
        "navigate":"NAVIGATE"
    }
    intent2id= {
        "restaurant":"name",
        "hotel":"name",
        "attraction":"name",
        "schedule":"event",
        "weather":"location",
        "navigate":"poi"
    }
    conn = sqlite3.connect(':memory:')
    cursor = conn.cursor()

    # query = input('User：')
    SQL_PROMPT_DICT = {
            "prompt_input": (
                "I want you to act as a SQL terminal in front of an example database. "
                "Below is an instruction that describes a task, Write a response that appropriately completes the request.\n\n"
                "###Instruction:\nBelow is the dialogue history and the Original question asked by user. Based on the history, you need to convert natural language question into SQL query.\n\n"
                "###History:\n{history}\n\n"
                "###Original question:\n{input}\n\n###Table Name:{table}\n\n###Table columns:{columns}\n\n###Example:{example}\n\n###Response: "
            ),
        }

    for mode in modes:
        for dataset in datasets:
            # with open("data/testdata/{}.json".format(dataset)) as f:
            with open("src/origindata/{}/{}.json".format(dataset,mode)) as f:
                data = json.load(f)
            # with open("FireflySQL/data/sftdata/{}_test.json".format(dataset)) as t:
            #     conversations = json.load(t)
            # assert len(data) == len(conversations)
            # print(len(data)) 
            # right = 0
            # wrong = 0
            # exact_right = 0
            # all = 0
            if dataset == "MultiWOZ":
                cursor.execute('''
                CREATE TABLE RESTAURANT (
                    name TEXT PRIMARY KEY,
                    food TEXT,    
                    address TEXT,
                    area TEXT,
                    phone TEXT,
                    postcode TEXT,
                    pricerange TEXT,
                    stars TEXT,
                    type TEXT,
                    choice TEXT,
                    ref TEXT
                )''')
                cursor.execute('''
                CREATE TABLE HOTEL (
                name TEXT PRIMARY KEY,
                address TEXT,
                area TEXT,
                phone TEXT,
                postcode TEXT,
                pricerange TEXT,
                stars TEXT,
                type TEXT,
                choice TEXT,
                ref TEXT,
                parking TEXT,
                interent TEXT
            )''')
            
                cursor.execute('''
                CREATE TABLE ATTRACTION (
                    name TEXT PRIMARY KEY,
                    address TEXT,
                    area TEXT,
                    phone TEXT,
                    postcode TEXT,
                    pricerange TEXT,
                    stars TEXT,
                    type TEXT,
                    choice TEXT,
                    ref TEXT
                )''')
            elif dataset =="CamRest":
                cursor.execute('''
                CREATE TABLE RESTAURANT (
                    id TEXT PRIMARY KEY,
                    name TEXT,
                    area TEXT,
                    food TEXT,
                    phone TEXT,
                    pricerange TEXT,
                    location TEXT,
                    address TEXT,
                    type TEXT,
                    postcode TEXT
                )''')
            else:
                cursor.execute('''
                CREATE TABLE SCHEDULE (
                    id Integer PRIMARY KEY,
                    event TEXT ,
                    time TEXT,    
                    date TEXT,
                    room TEXT,
                    agenda TEXT,
                    party TEXT
                )''')
                cursor.execute('''
                CREATE TABLE WEATHER (
                    id Integer PRIMARY KEY,
                    location TEXT  ,
                    monday TEXT,    
                    tuesday TEXT,
                    wednesday TEXT,
                    thursday TEXT,
                    friday TEXT,
                    saturday TEXT,
                    sunday TEXT,
                    today TEXT
                )''')
                cursor.execute('''
                CREATE TABLE NAVIGATE (
                    id Integer PRIMARY KEY,
                    poi TEXT  ,
                    poi_type TEXT,    
                    address TEXT,
                    distance TEXT,
                    traffic_info TEXT
                )''')

            new_dialogues = []
            for id,sample in enumerate(tqdm(data,position=0)):
                # print(sample)
                dialogue = sample["dialogue"]
                scenario = sample["scenario"]
                items = scenario["kb"]["items"]
                if items == None:
                    
                    new_dialogues.append(sample)
                    continue
                if dataset == "CamRest":
                    items = remove_duplicates_by_id(items)    
                
                example = items[0]
                category = scenario["task"]["intent"]
                if dataset == "MultiWOZ":
                    if category not in ["restaurant","attraction","hotel"]:
                        continue
                column_names = sample["scenario"]["kb"]["column_names"]
                if items == None:
                    continue
                # print(id)
                for item in items:
                    keys = item.keys()
                    values = tuple(item[key] for key in keys)
                    cursor.execute('''
                        INSERT INTO {} ({}) VALUES ({})
                        '''.format(intent2table[category],', '.join(keys), ', '.join(['?']*len(values))), values)
                conn.commit()
                history = ""
                for utterance in dialogue:
                    if utterance["turn"] == "user":
                        user_utter = utterance["utterance"]
                        
                        continue
                    if utterance["turn"] == "system":
                        sys_utter = utterance["utterance"]
                        annotated_query = utterance["annotated_query"]
                        annotated_knowledge = utterance["annotated_knowledge"]

                        # all += 1
                        # print(all)
                        query = SQL_PROMPT_DICT["prompt_input"].format(input=user_utter,columns=column_names,table=category,example = json.dumps(example),history=history)
                        history = history +" User: "+ user_utter+ "\n"
                        history = history + " Assistant: " + sys_utter + "\n\n"
                        query = query.strip()
                        input_ids = build_prompt(tokenizer, template, query, [], system=None).to(model.device)
                        outputs = model.generate(
                            input_ids=input_ids, max_new_tokens=max_new_tokens, do_sample=True,
                            top_p=top_p, temperature=temperature, repetition_penalty=repetition_penalty,
                            eos_token_id=stop_token_id
                        )
                        outputs = outputs.tolist()[0][len(input_ids[0]):]
                        response = tokenizer.decode(outputs)
                        response = response.strip().replace(template.stop_word, "").strip()
                        # print("repsonse:{}".format(response))
                        # exit()
                        
                        if annotated_query == "[NOTHING]":
                            # print(response)
                            if response =="" or response == "Unable to convert query to SQL statement.":
                                utterance["sql_extract"] = "exact_right"
                            else:
                                utterance["sql_extract"] = "wrong"
                            continue
                        if response.startswith("SELECT"):
                            response = response.replace("CAMREST","RESTAURANT")
                            try:
                                cursor.execute(response)
                                row = cursor.fetchall()
                            except:
                                utterance["sql_extract"] = "wrong"
                                # wrong +=1
                                continue
                            
                            if dataset == "MultiWOZ":
                                sql_id_list = [t[0] for t in row]
                                annotated_id_list = [d["name"] for d in annotated_knowledge if d["label"] == '1']
                                new_annotated_knowlegde =[]
                                for d in annotated_knowledge:
                                    if d["name"] in sql_id_list:
                                        d["label"] = "1"
                                    else:
                                        d["label"] = "0"
                                    new_annotated_knowlegde.append(d)
                                utterance["annotated_knowledge"] = new_annotated_knowlegde
                            elif dataset =="CamRest":
                                sql_id_list = [t[0] for t in row]
                                annotated_id_list = [d["id"] for d in annotated_knowledge if d["label"] == '1']
                                new_annotated_knowlegde =[]
                                for d in annotated_knowledge:
                                    if d["id"] in sql_id_list:
                                        d["label"] = "1"
                                    else:
                                        d["label"] = "0"
                                    new_annotated_knowlegde.append(d)
                                utterance["annotated_knowledge"] = new_annotated_knowlegde
                            else:
                                sql_id_list = [t[1:] for t in row]
                                # print(row)
                                # print(annotated_knowledge)
                                
                                sql_id_list = [tuple(sorted(t)) for t in sql_id_list]
                                # print(sql_id_list)

                                # annotated_id_list = [index+1 for index,d in enumerate(annotated_knowledge) if d["label"] == '1']
                                annotated_id_list = []
                                new_annotated_knowlegde =[]
                                for knowledge in annotated_knowledge:
                                    tuple_of_values = tuple(v for k, v in knowledge.items() if k != 'label')
                                    tuple_of_values = sorted(tuple_of_values)
                                    if knowledge["label"] == "1":
                                        annotated_id_list.append(tuple(tuple_of_values))
                                    if(tuple(tuple_of_values) in sql_id_list):
                                        knowledge["label"] = "1"
                                        new_annotated_knowlegde.append(knowledge)
                                    else:
                                        knowledge["label"] = "0"
                                        new_annotated_knowlegde.append(knowledge)
                                utterance["annotated_knowledge"] = new_annotated_knowlegde
                                # print(annotated_id_list)
                                # print(new_annotated_knowlegde)
                                # exit()
                            difference_ids_1 = set(annotated_id_list) - set(sql_id_list)
                            difference_ids_2 = set(sql_id_list) - set(annotated_id_list)
                            # print(sql_id_list)
                            # print(annotated_id_list)
                            utterance["response"] = response
                            if len(difference_ids_1) == 0:
                                # right += 1
                                if len(difference_ids_2) ==0:
                                    # exact_right += 1
                                    utterance["sql_extract"] = "exact_right"
                                else:
                                    utterance["sql_extract"] = "right"
                            else:
                                # wrong += 1  
                                utterance["sql_extract"] = "wrong"
                        else:
                            #! 如果response有问题时，不保存response
                            # wrong += 1
                            utterance["sql_extract"] = "wrong"
                cursor.execute('DELETE from {};'.format(intent2table[category]))   
                new_dialogues.append(sample)
            # print(all)
            # print(right) 
            # print(exact_right)
            # print(wrong)
            # print(right/all)
            # print(exact_right/all)

            
            os.makedirs("src/sql/test/{}_output_{}".format(template_name,trainsql_version),exist_ok=True)
            with open("src/sql/test/{}_output_{}/{}_{}.json".format(template_name,trainsql_version,dataset,mode),"w") as f:
                json.dump(new_dialogues,f)
            # cursor.execute("DROP TABLE  IF EXISTS RESTAURANT")
            if dataset=="MultiWOZ":
                cursor.execute("DROP TABLE  IF EXISTS RESTAURANT")          
                cursor.execute("DROP TABLE  IF EXISTS HOTEL")             
                cursor.execute("DROP TABLE  IF EXISTS ATTRACTION")       
            elif dataset == "CamRest":
                cursor.execute("DROP TABLE  IF EXISTS RESTAURANT")       
            else:
                cursor.execute("DROP TABLE  IF EXISTS SCHEDULE")       
                cursor.execute("DROP TABLE  IF EXISTS NAVIGATE")       
                cursor.execute("DROP TABLE  IF EXISTS WEATHER")    


if __name__ == '__main__':
    main()

