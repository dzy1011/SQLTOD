import json
import sys
import os
import jsonlines
import argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# sys.path.append("../../")
# dataset = "CamRest"
# dataset = "MultiWOZ"

def batch_generate_sql_train_data():
    datasets = ["CamRest","MultiWOZ","SMD"]
    modes = ["train"]
    # sql_version = "exactsqlv3-2"
    parser = argparse.ArgumentParser()
    # parser.add_argument("--sql_version", type=str)
    parser.add_argument("--mode", type=str, default="train")
    # parser.add_argument("--dataset")
    # parser.add_argument("--mode", type=str)
    args = parser.parse_args()
    # sql_version = args.sql_version
    mode = args.mode
    for mode in modes:
        conversations = []
        conversation_id = 0
        for dataset in datasets:
            
            # filepath = "data/{}/sql/{}.json".format(dataset,mode)
            # filepath = "origindata/updatesql/update{}/{}_{}.json".format(sql_version,dataset,mode)
            filepath = "final_sql/{}_{}.json".format(dataset,mode)

            with open(filepath,"r") as f:
                data = json.load(f)
            SQL_PROMPT_DICT = {
                "prompt_input": (
                    "I want you to act as a SQL terminal in front of an example database. "
                    "Below is an instruction that describes a task, Write a response that appropriately completes the request.\n\n"
                    "###Instruction:\nBelow is the dialogue history and the Original question asked by user. Based on the history, you need to convert natural language question into SQL query.\n\n"
                    "###History:\n{history}\n\n"
                    "###Original question:\n{input}\n\n###Table Name:{table}\n\n###Table columns:{columns}\n\n###Example:{example}\n\n###Response: "
                ),
                "prompt_no_input": (
                    "I want you to act as a SQL terminal in front of an example database. "
                    "Below is an instruction that describes a task, Write a response that appropriately completes the request.\n\n"
                    "###Instruction:\n{instruction}\n\n### Response: "
                ),
            }
            instruction = ""
            temp_conversations = []
            # conversation_id = 0
            # conversation_id = 1666
            # conversation_id = 10085
            # temp_conversation_id = 0
            for sample in data:
                dialogue = sample["dialogue"]
                scenario = sample["scenario"]
                column_names = scenario["kb"]["column_names"]
                catergory = scenario["task"]["intent"]
                if scenario["kb"]["items"] == None:
                    continue
                example= scenario["kb"]["items"][0]
                history = ""
                for turn in dialogue:
                    if turn["turn"] == "user":
                        user_utter = turn["utterance"]
                        
                        continue
                    else:
                        sys_utter = turn["utterance"]
                        # anno_query = turn["annotated_query"]
                        sql = turn["sql"]
                        if sql == "":
                            sql = "Unable to convert query to SQL statement."
                            break
                        # if sql_version== "v2":
                        #     sql = turn["full_sql"]
                        # if sql_version in ["v1-2","_exactv1-2"]:
                        #     sql=turn["update_sql"]
                    conversation_id += 1
                    conversation = {"conversation_id":conversation_id,
                                "catergory":catergory,
                                "conversation":[{"human":SQL_PROMPT_DICT["prompt_input"].format(input=user_utter,columns=column_names,table=catergory,example = json.dumps(example),history=history),"assistant":sql}],
                                "dataset":dataset}
                    history = history +" User: "+ user_utter+ "\n"
                    history = history + " Assistant: " + sys_utter + "\n\n"
                    temp_conversations.append(conversation)
            conversations = conversations+temp_conversations
            os.makedirs("src/sql/sftfinal_sql",exist_ok=True)
            with jsonlines.open("src/sql/sftfinal_sql/{}_{}.jsonl".format(dataset,mode),"w") as f:
                for conversation in temp_conversations:
                    f.write(conversation)
        with jsonlines.open("src/sql/sftfinal_sql/all_{}.jsonl".format(mode),"w") as w:
            for conversation in conversations:
                w.write(conversation)


def SMD_generate_zero_shot_sql(filepath, outputpath):
    conversations = []
    conversation_id = 0
    
        
    # filepath = "data/{}/sql/{}.json".format(dataset,mode)
    # filepath = "origindata/updatesql/update{}/{}_{}.json".format(sql_version,dataset,mode)
    # filepath = "origindata/new_updatesql/{}/{}_{}.json".format(sql_version,dataset,mode)
    with open(filepath,"r") as f:
        data = json.load(f)
    SQL_PROMPT_DICT = {
        "prompt_input": (
            "I want you to act as a SQL terminal in front of an example database. "
            "Below is an instruction that describes a task, Write a response that appropriately completes the request.\n\n"
            "###Instruction:\nBelow is the dialogue history and the Original question asked by user. Based on the history, you need to convert natural language question into SQL query.\n\n"
            "###History:\n{history}\n\n"
            "###Original question:\n{input}\n\n###Table Name:{table}\n\n###Table columns:{columns}\n\n###Example:{example}\n\n###Response: "
        ),
        "prompt_no_input": (
            "I want you to act as a SQL terminal in front of an example database. "
            "Below is an instruction that describes a task, Write a response that appropriately completes the request.\n\n"
            "###Instruction:\n{instruction}\n\n### Response: "
        ),
    }
    instruction = ""
    temp_conversations = []
    # conversation_id = 0
    # conversation_id = 1666
    # conversation_id = 10085
    # temp_conversation_id = 0
    for sample in data:
        dialogue = sample["dialogue"]
        scenario = sample["scenario"]
        column_names = scenario["kb"]["column_names"]
        catergory = scenario["task"]["intent"]
        if scenario["kb"]["items"] == None:
            continue
        example= scenario["kb"]["items"][0]
        history = ""
        for turn in dialogue:
            if turn["turn"] == "user":
                user_utter = turn["utterance"]
                
                continue
            else:
                sys_utter = turn["utterance"]
                # anno_query = turn["annotated_query"]
                sql = turn["sql"]
                if sql == "":
                    sql = "Unable to convert query to SQL statement."
                    break
                # if sql_version== "v2":
                #     sql = turn["full_sql"]
                # if sql_version in ["v1-2","_exactv1-2"]:
                #     sql=turn["update_sql"]
            conversation_id += 1
            conversation = {"conversation_id":conversation_id,
                        "catergory":catergory,
                        "conversation":[{"human":SQL_PROMPT_DICT["prompt_input"].format(input=user_utter,columns=column_names,table=catergory,example = json.dumps(example),history=history),"assistant":sql}],
            }
            history = history +" User: "+ user_utter+ "\n"
            history = history + " Assistant: " + sys_utter + "\n\n"
            temp_conversations.append(conversation)
    conversations = conversations+temp_conversations
    # os.makedirs("origindata/sftdata{}".format(sql_version),exist_ok=True)
    with jsonlines.open(outputpath,"w") as f:
        for conversation in temp_conversations:
            f.write(conversation)

if __name__=="__main__":
    # filepath = "origindata/SMD_zero-shot/sch_wea_train.json"
    # outputpath = "origindata/SMD_zero-shot/sftdata/sch_wea_train.jsonl"
    # SMD_generate_zero_shot_sql(filepath, outputpath)
    batch_generate_sql_train_data()