import json
import random
import os
from utils import linearize_knowledge, preprocess_text
import argparse
# inp_file = json.load(open('origindata/MultiWOZ/test.json'))

# inp_file = json.load(open('origindata/SMD_zero-shot/sch_wea_train.json'))
# os.makedirs("dialogue/train_data-zero-shot/dial",exist_ok=True)
# out_file = open("dialogue/train_data-zero-shot/dial/sch_wea_train-80.jsonl", "w", encoding = 'utf-8')


# inp_file = json.load(open('origindata/CamRest/train.json'))
# os.makedirs("dialogue/train_data-80-linear",exist_ok=True)
# out_file = open("dialogue/train_data-80-linear/CamRest_train.jsonl", "w", encoding = 'utf-8')

parser = argparse.ArgumentParser()
parser.add_argument("--inp_file", type=str)
parser.add_argument("--out_file", type=str)
args = parser.parse_args()
inp_file = args.inp_file 
inp_file = json.load(open(inp_file))
out_file = args.out_file
os.makedirs(os.path.dirname(out_file), exist_ok=True)
out_file = open(out_file, "w", encoding = 'utf-8')

window = 3
select_list = [1,2,4,6,8]

RESONSE_PROMPT_DICT = {
            "prompt_input": (
                "I want you to act as a personal assistant who needs to answer user questions based on the knowledge base. "
                "Below is an instruction that describes a task, Write a response that appropriately completes the request.\n\n"
                "###Instruction:\nBelow is the dialogue history and the question asked by user. Based on the history, you need to answer user questions based on the knowledge base.\n\n"
                "###Knowledge base:\n{knowledge}\n\n"
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
    select_rand = random.choice(select_list)
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

            if annotated_knowledge != []:
                retrieved_konwledge = []
                
                

                if select_rand%2==0:
                    for kb in annotated_knowledge:
                        if kb["label"] == "1":
                            kb.pop("label")
                            retrieved_konwledge.append(kb)
            #                 knowledge += json.dumps(kb)
                
                else:
                    for kb in annotated_knowledge:
                        kb.pop("label")
                        retrieved_konwledge.append(kb)
            #             knowledge += json.dumps(kb)
                knowledge = linearize_knowledge(retrieved_konwledge,fields)

                # print(retrieved_knowledge_seq)
                # exit()
            else:
                knowledge = 'no knowledge base'
            #tmp_dia['human'] = 'You are a personal assistant who needs to answer user questions based on the knowledge base. ' + 'knowledge base: '+ knowledge + ' user: ' + ' '.join(dia_history)
            # tmp_dia['human'] = RESONSE_PROMPT_DICT["prompt_input"].format(knowledge=knowledge,input=dia_history[-1], history=' '.join(dia_history[:-1]))
            # tmp_dia['human'] = RESONSE_PROMPT_DICT["prompt_input"].format(knowledge=knowledge,input=dia_history[-1], history=' '.join(dia_history[-window-1:-1]))
            # print(preprocess_text(dia_history[-1]))
            # print(preprocess_text(' '.join(dia_history[:-1])))
            # exit()
            tmp_dia['human'] = RESONSE_PROMPT_DICT["prompt_input"].format(knowledge=knowledge,input=preprocess_text(dia_history[-1]), history=preprocess_text(' '.join(dia_history[:-1])))
            # print(tmp_dia["human"])
            # exit()
            tmp_dia['assistant'] = system
            dia_history.append("assistant: "+system)
            res["conversation"].append(tmp_dia)
            json.dump(res, out_file, ensure_ascii=False)
            out_file.write('\n')
            tmp_dia = {}
            res["conversation"] = []


        
        

    


