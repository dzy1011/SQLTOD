import json
# import random
import os
import argparse
import string

def preprocess_text(text):
    """Preprocess utterance and table value."""
    text = text.strip().replace("\t", " ").lower()
    for p in string.punctuation:
        text = text.replace(p, f" {p} ")
    text = " ".join(text.split())
    return text


def linearize_knowledge_record(knowledge_record, fields):
    """Convert a knowledge record into a flatten sequence with special symbols."""
    knowledge_seq = []
    for f in fields:
        value = preprocess_text(str(knowledge_record.get(f, "")))
        knowledge_seq.append(f.replace("_", " ") + " : " + value)
    return " | ".join(knowledge_seq)


def linearize_knowledge(knowledge, fields):
    """Convert knowledge into a flatten sequecen with special symbols."""
    knowledge_seq = []
    knowledge_seq.append("col : " + " | ".join(map(lambda x: x.replace("_", " "), fields)))
    for idx, record in enumerate(knowledge):
        values = []
        for f in fields:
            v = preprocess_text(str(record.get(f, "")))
            values.append(v)

        record_seq = " | ".join(values)
        knowledge_seq.append(f"row {idx} : {record_seq}")
    return " || ".join(knowledge_seq)
def kb_to_conv_new_prompt_linear(inp_file,out_file):
    inp_file = json.load(open(inp_file))
    out_file = open(out_file, "w", encoding = 'utf-8')

    window = 3
    select_list = [1,2]

    RESONSE_PROMPT_DICT = {
                "prompt_input": (
                    "I want you to act as a personal assistant who needs to answer user questions based on the knowledge base. "
                    "Below is an instruction that describes a task, Write a response that appropriately completes the request.\n\n"
                    "###Instruction:\nBelow is the dialogue history and the question asked by user. Based on the history, you need to answer user questions based on the knowledge base.\n\n"
                    "###Knowledge base:\n{knowledge}\n\n"
                    "###History:\n{history}\n\n"
                    "###Question:\n{input}\n\n###Response: "
                ),
                "prompt_input_with_sql": (
                    "I want you to act as a personal assistant who needs to answer user questions based on the knowledge base. "
                    "Below is an instruction that describes a task, Write a response that appropriately completes the request.\n\n"
                    "###Instruction:\nBelow is the dialogue history and the question asked by user. Based on the history, you need to answer user questions based on the knowledge base.\n\n"
                    "###Knowledge base:\n{knowledge}\n\n"
                    "###SQL:\n{SQL}\n\n"
                    "###History:\n{history}\n\n"
                    "###Question:\n{input}\n\n###Response: "
            ),
                "prompt_no_input": (
                    "I want you to act as a personal assistant who needs to answer user questions based on the knowledge base. "
                    "Below is an instruction that describes a task, Write a response that appropriately completes the request.\n\n"
                    "###Instruction:\n{instruction}\n\n### Response: "
                ),
            }

    for idx, dial in enumerate(inp_file):
        dia_history = []
        res={}
        res["conversation_id"] = idx
        res["category"] = 'dialog'
        res["conversation"] = []
        tmp_dia = {}
        # select_rand = random.choice(select_list)
        excluded_fields = []
        fields = [name for name in dial["scenario"]["kb"]["column_names"] if name not in excluded_fields]
        for turns in range(len(dial['dialogue'])):
            if dial['dialogue'][turns]['turn'] == "user":
                dia_history.append("user: "+dial['dialogue'][turns]['utterance'])
            else:
                system = dial['dialogue'][turns]['utterance']
                knowledge = ''
                # print("dial['dialogue'][turns]",dial['dialogue'][turns])
                annotated_knowledge = dial['dialogue'][turns]['annotated_knowledge']
                sql = dial["dialogue"][turns].get("response","")
                if annotated_knowledge != []:
                    retrieved_konwledge = []
                # if select_rand%2==0:
                    for kb in annotated_knowledge:
                        if kb["label"] == "1":
                            kb.pop("label")
                            retrieved_konwledge.append(kb)
                            # knowledge += json.dumps(kb)
                    knowledge = linearize_knowledge(retrieved_konwledge,fields)
                    # else:
                    #     for kb in annotated_knowledge:
                    #         kb.pop("label")
                    #         knowledge += json.dumps(kb)
                else:
                    knowledge = 'no knowledge base'
                #tmp_dia['human'] = 'You are a personal assistant who needs to answer user questions based on the knowledge base. ' + 'knowledge base: '+ knowledge + ' user: ' + ' '.join(dia_history)
                tmp_dia['human'] = RESONSE_PROMPT_DICT["prompt_input"].format(knowledge=knowledge,input=dia_history[-1], history=' '.join(dia_history[:-1]))
                # tmp_dia['human'] = RESONSE_PROMPT_DICT["prompt_input_with_sql"].format(knowledge=knowledge,input=dia_history[-1], history=' '.join(dia_history[-window-1:-1]),SQL=sql)
                tmp_dia['assistant'] = system
                dia_history.append("assistant: "+system)
                res["conversation"].append(tmp_dia)
                json.dump(res, out_file, ensure_ascii=False)
                out_file.write('\n')
                tmp_dia = {}
                res["conversation"] = []

def kb_to_conv_old_prompt(inp_file, out_file):
    inp_file = json.load(open(inp_file))
    out_file = open(out_file, "w", encoding = 'utf-8')

    window = 2

    for idx, dial in enumerate(inp_file):
        dia_history = []
        res={}
        res["conversation_id"] = idx
        res["category"] = 'dialog'
        res["conversation"] = []
        tmp_dia = {}


        for turns in range(len(dial['dialogue'])):
            if dial['dialogue'][turns]['turn'] == "user":
                dia_history.append(dial['dialogue'][turns]['utterance'])
            else:
                system = dial['dialogue'][turns]['utterance']
                knowledge = ''
                annotated_knowledge = dial['dialogue'][turns]['annotated_knowledge']
                if annotated_knowledge != []:
                    for kb in annotated_knowledge:
                        if kb["label"] == '1':
                            knowledge += json.dumps(kb)
                else:
                    knowledge = 'no knowledge base'
                tmp_dia['human'] = 'You are a personal assistant who needs to answer user questions based on the knowledge base. ' + 'knowledge base: '+ knowledge + ' user: ' + ' '.join(dia_history)
                tmp_dia['assistant'] = system
                dia_history.append(system)
                res["conversation"].append(tmp_dia)
                json.dump(res, out_file, ensure_ascii=False)
                out_file.write('\n')
                tmp_dia = {}
                res["conversation"] = []

def kb_to_conv_new_prompt(inp_file,out_file):
    inp_file = json.load(open(inp_file))
    out_file = open(out_file, "w", encoding = 'utf-8')

    window = 3
    select_list = [1,2]

    RESONSE_PROMPT_DICT = {
                "prompt_input": (
                    "I want you to act as a personal assistant who needs to answer user questions based on the knowledge base. "
                    "Below is an instruction that describes a task, Write a response that appropriately completes the request.\n\n"
                    "###Instruction:\nBelow is the dialogue history and the question asked by user. Based on the history, you need to answer user questions based on the knowledge base.\n\n"
                    "###Knowledge base:\n{knowledge}\n\n"
                    "###History:\n{history}\n\n"
                    "###Question:\n{input}\n\n###Response: "
                ),
                "prompt_input_with_sql": (
                    "I want you to act as a personal assistant who needs to answer user questions based on the knowledge base. "
                    "Below is an instruction that describes a task, Write a response that appropriately completes the request.\n\n"
                    "###Instruction:\nBelow is the dialogue history and the question asked by user. Based on the history, you need to answer user questions based on the knowledge base.\n\n"
                    "###Knowledge base:\n{knowledge}\n\n"
                    "###SQL:\n{SQL}\n\n"
                    "###History:\n{history}\n\n"
                    "###Question:\n{input}\n\n###Response: "
            ),
                "prompt_no_input": (
                    "I want you to act as a personal assistant who needs to answer user questions based on the knowledge base. "
                    "Below is an instruction that describes a task, Write a response that appropriately completes the request.\n\n"
                    "###Instruction:\n{instruction}\n\n### Response: "
                ),
            }

    for idx, dial in enumerate(inp_file):
        dia_history = []
        res={}
        res["conversation_id"] = idx
        res["category"] = 'dialog'
        res["conversation"] = []
        tmp_dia = {}
        # select_rand = random.choice(select_list)

        for turns in range(len(dial['dialogue'])):
            if dial['dialogue'][turns]['turn'] == "user":
                dia_history.append("user: "+dial['dialogue'][turns]['utterance'])
            else:
                system = dial['dialogue'][turns]['utterance']
                knowledge = ''
                # print("dial['dialogue'][turns]",dial['dialogue'][turns])
                annotated_knowledge = dial['dialogue'][turns]['annotated_knowledge']
                sql = dial["dialogue"][turns].get("response","")
                if annotated_knowledge != []:
                    # if select_rand%2==0:
                        for kb in annotated_knowledge:
                            if kb["label"] == "1":
                                kb.pop("label")
                                knowledge += json.dumps(kb)
                    # else:
                    #     for kb in annotated_knowledge:
                    #         kb.pop("label")
                    #         knowledge += json.dumps(kb)
                else:
                    knowledge = 'no knowledge base'
                #tmp_dia['human'] = 'You are a personal assistant who needs to answer user questions based on the knowledge base. ' + 'knowledge base: '+ knowledge + ' user: ' + ' '.join(dia_history)
                tmp_dia['human'] = RESONSE_PROMPT_DICT["prompt_input"].format(knowledge=knowledge,input=dia_history[-1], history=' '.join(dia_history[-window-1:-1]))
                # tmp_dia['human'] = RESONSE_PROMPT_DICT["prompt_input_with_sql"].format(knowledge=knowledge,input=dia_history[-1], history=' '.join(dia_history[-window-1:-1]),SQL=sql)
                tmp_dia['assistant'] = system
                dia_history.append("assistant: "+system)
                res["conversation"].append(tmp_dia)
                json.dump(res, out_file, ensure_ascii=False)
                out_file.write('\n')
                tmp_dia = {}
                res["conversation"] = []


   
if __name__ == "__main__":
        parser = argparse.ArgumentParser()
        # parser.add_argument("--train_args_file", type=str, default='train_args/pretrain/full/bloom-1b1-pretrain-full.json', help="")
        parser.add_argument("--kb_version", type=str, default='', help="")
        parser.add_argument("--window_size", type=str, default='')
        parser.add_argument("--zero_shot", action="store_true",)
        parser.add_argument("--intent", type=str, default="")
        args = parser.parse_args()
        kb_version = args.kb_version
        window_size = args.window_size
        zero_shot = args.zero_shot
        intent = args.intent
        
        if zero_shot:
            datasets = ["SMD"]
            modes = ["test"]
            models = ["qwen"]
            # prompts = ["new_prompt","old_prompt"]
            prompts = ["new_prompt"]
            for prompt in prompts:
                for dataset in datasets:
                    for mode in modes:
                        for model in models:
                            inp_file = "dialogue/generate_data/small_kb-{}/{}/{}_{}_{}.json".format(kb_version,model,dataset, mode,intent[:3])
                            os.makedirs("dialogue/generate_data/small_kb-{}{}/{}/{}/conversations".format(kb_version,window_size,model,prompt),exist_ok=True)
                            out_file = "dialogue/generate_data/small_kb-{}{}/{}/{}/conversations/{}_{}_{}.jsonl".format(kb_version,window_size,model,prompt,dataset, mode,intent[:3])
                            if prompt == "new_prompt":
                                # kb_to_conv_new_prompt(inp_file,out_file)
                                kb_to_conv_new_prompt_linear(inp_file, out_file)
                            elif prompt == "old_prompt":
                                kb_to_conv_old_prompt(inp_file,out_file)
        else:
            datasets = ["CamRest","MultiWOZ","SMD"]
            modes = ["test"]
            models = ["qwen",]
            # prompts = ["new_prompt","old_prompt"]
            prompts = ["new_prompt"]
            for prompt in prompts:
                for dataset in datasets:
                    for mode in modes:
                        for model in models:
                            inp_file = "dialogue/generate_data/small_kb-{}/{}/{}_{}.json".format(kb_version,model,dataset, mode)
                            os.makedirs("dialogue/generate_data/small_kb-{}{}/{}/{}/conversations".format(kb_version,window_size,model,prompt),exist_ok=True)
                            out_file = "dialogue/generate_data/small_kb-{}{}/{}/{}/conversations/{}_{}.jsonl".format(kb_version,window_size,model,prompt,dataset, mode)
                            if prompt == "new_prompt":
                                # kb_to_conv_new_prompt(inp_file,out_file)
                                kb_to_conv_new_prompt_linear(inp_file, out_file)
                            elif prompt == "old_prompt":
                                kb_to_conv_old_prompt(inp_file,out_file)

            
            # datasets = ["CamRest","MultiWOZ"]
            # modes = ["test","train"]
            # models = ["qwen",]
            # # prompts = ["new_prompt","old_prompt"]
            # prompts = ["new_prompt"]

            # for prompt in prompts:
            #     for dataset in datasets:
            #         for mode in modes:
            #             for model in models:
            #                 inp_file = "dialogue/generate_data/large_kbv2/{}/{}_{}.json".format(model,dataset, mode)
            #                 os.makedirs("dialogue/generate_data/large_kbv2/{}/{}/conversations".format(model,prompt),exist_ok=True)
            #                 out_file = "dialogue/generate_data/large_kbv2/{}/{}/conversations/{}_{}.jsonl".format(model,prompt,dataset, mode)
            #                 if prompt == "new_prompt":
            #                     kb_to_conv_new_prompt(inp_file,out_file)
            #                 elif prompt == "old_prompt":
            #                     kb_to_conv_old_prompt(inp_file,out_file)
            

        


