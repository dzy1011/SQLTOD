import json
import sqlite3
import os
from copy import deepcopy
import re
import argparse

def remove_duplicates_by_id(items):
    unique_items = []
    seen_ids = set()
    for item in items:
        if item['id'] not in seen_ids:
            unique_items.append(item)
            seen_ids.add(item['id'])
    return unique_items

def remove_duplicate_dicts(dict_list):
    """
    Remove duplicate dictionaries from a list.

    Args:
        dict_list (list): The list of dictionaries to check for duplicates.

    Returns:
        list: The list of dictionaries with duplicates removed.
    """

    unique_dicts = []
    seen_dicts = set()
    for dict in dict_list:
        dict_hash = hash(tuple(dict.items()))  # Create a hash of the dictionary
        if dict_hash not in seen_dicts:
            unique_dicts.append(dict)
            seen_dicts.add(dict_hash)

    return unique_dicts
def extract_small_kb_by_sql(origindatapath, modeloutputpath, writedatapath, dataset):

    with open(origindatapath,"r") as f:
        origindata = json.load(f)
    with open(modeloutputpath,"r") as f: 
        outputdata = json.load(f)
    assert len(origindata) == len(outputdata)
    intent2table = {
        "restaurant":"RESTAURANT",
        "schedule":"SCHEDULE",
        "attraction":"ATTRACTION",
        "hotel":"HOTEL",
        "weather":"WEATHER",
        "navigate":"NAVIGATE"
    }
    conn = sqlite3.connect(':memory:')
    cursor = conn.cursor()
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
    new_samples = []
    for id,(originsample, outputsample) in enumerate(zip(origindata,outputdata)):
        # newsample = deepcopy(originsample)
        scenario = originsample["scenario"]
        items = scenario["kb"]["items"]
        if dataset == "CamRest":
            items = remove_duplicates_by_id(items)   
        category = scenario["task"]["intent"]
        origindialogue = originsample["dialogue"]
        outputdialogue = outputsample["dialogue"]

        assert len(origindialogue) == len(outputdialogue)

        if items ==  None:
            new_samples.append(originsample)
            continue
        for item in items:
            keys = item.keys()
            values = tuple(item[key] for key in keys)
            cursor.execute('''
                INSERT INTO {} ({}) VALUES ({})
                '''.format(intent2table[category],', '.join(keys), ', '.join(['?']*len(values))), values)
        conn.commit()
        for utter_id, utterance in enumerate(origindialogue):
            if utterance["turn"] == "user":
                continue
            if utterance["turn"] == "system":
                
                output_utterance = outputdialogue[utter_id]
                annotated_query = utterance["annotated_query"]
                annotated_knowledge = utterance["annotated_knowledge"]
                del utterance["sql"]
                if "response" not in output_utterance:
                    # 说明模型生成的SQL格式可能有问题，无法提取知识
                    # 将annotated_knowledge 设置为空
                    utterance["response"] = ""
                    utterance["sql_extract"] = output_utterance["sql_extract"]
                    utterance["annotated_knowledge"] = []
                    
                else:
                    # print()
                    response = output_utterance["response"]
                    utterance["response"] = response
                    utterance["sql_extract"] = output_utterance["sql_extract"]
                    cursor.execute(response)
                    row = cursor.fetchall()
                    if dataset == "MultiWOZ":
                        sql_id_list = [t[0] for t in row]
                        # annotated_id_list = [d["name"] for d in annotated_knowledge if d["label"] == '1']
                        new_annotated_knowlegde =[]
                        for d in annotated_knowledge:
                            if d["name"] in sql_id_list:
                                d["label"] = "1"
                            # else:
                            #     d["label"] = "0"
                                new_annotated_knowlegde.append(d)
                        utterance["annotated_knowledge"] = new_annotated_knowlegde
                    elif dataset =="CamRest":
                        sql_id_list = [t[0] for t in row]
                        # annotated_id_list = [d["id"] for d in annotated_knowledge if d["label"] == '1']
                        new_annotated_knowlegde =[]
                        for d in annotated_knowledge:
                            if d["id"] in sql_id_list:
                                d["label"] = "1"
                            # else:
                            #     d["label"] = "0"
                                new_annotated_knowlegde.append(d)
                        utterance["annotated_knowledge"] = new_annotated_knowlegde
                    else:
                        sql_id_list = [t[1:] for t in row]
                        sql_id_list = [tuple(sorted(t)) for t in sql_id_list]

                        # annotated_id_list = []
                        new_annotated_knowlegde =[]
                        for knowledge in annotated_knowledge:
                            tuple_of_values = tuple(v for k, v in knowledge.items() if k != 'label')
                            tuple_of_values = sorted(tuple_of_values)
                            # if knowledge["label"] == "1":
                                # annotated_id_list.append(tuple(tuple_of_values))
                            if(tuple(tuple_of_values) in sql_id_list):
                                knowledge["label"] = "1"
                                new_annotated_knowlegde.append(knowledge)
                            # else:
                            #     knowledge["label"] = "0"
                            #     new_annotated_knowlegde.append(knowledge)
                        utterance["annotated_knowledge"] = new_annotated_knowlegde

        cursor.execute('DELETE from {};'.format(intent2table[category]))     
        new_samples.append(originsample)          
        # os.makedirs("script/newtest/{}_output_{}".format(template_name,trainsql_version),exist_ok=True)
    with open(writedatapath,"w") as f:
        json.dump(new_samples,f)
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

def extract_large_kb_by_sql(origindatapath, modeloutputpath, writedatapath, dataset):
    #! large 
    with open(origindatapath,"r") as f:
        origindata = json.load(f)
    with open(modeloutputpath,"r") as f: 
        outputdata = json.load(f)
    assert len(origindata) == len(outputdata)
    intent2table = {
        "restaurant":"RESTAURANT",
        "schedule":"SCHEDULE",
        "attraction":"ATTRACTION",
        "hotel":"HOTEL",
        "weather":"WEATHER",
        "navigate":"NAVIGATE"
    }
    conn = sqlite3.connect(':memory:')
    cursor = conn.cursor()
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
    new_samples = []
    #! 构建大知识库
    if dataset!= "SMD":
        all_items = []
        all_names = []
        for id, sample in enumerate(origindata):
            dialogue = sample["dialogue"]
            scenario = sample["scenario"]
            items = scenario["kb"]["items"]
            if dataset == "CamRest":
                items = remove_duplicates_by_id(items)    
            category = scenario["task"]["intent"]
            if dataset == "MultiWOZ":
                if category not in ["restaurant","attraction","hotel"]:
                    continue

            for item in items:
                # if item not in all_items:
                if item["name"] not in all_names:
                    # print(item["name"])
                    keys = item.keys()
                    values = tuple(item[key] for key in keys)
                    cursor.execute('''
                        INSERT INTO {} ({}) VALUES ({})
                        '''.format(intent2table[category],', '.join(keys), ', '.join(['?']*len(values))), values)
                    all_items.append(item)
                    all_names.append(item["name"])
        assert len(all_items) == len(all_names)
        print(len(all_items))
        # all_items = remove_duplicate_dicts(all_items)
        # print(len(all_items))
        # exit()
    else:
        # ! 如果是SMD
        all_items = []
        for id,sample in enumerate(origindata):
            dialogue = sample["dialogue"]
            scenario = sample["scenario"]
            items = scenario["kb"]["items"]
            category = scenario["task"]["intent"]
            if items==None:
                continue
            for item in items:
                if item not in all_items:
                    keys = item.keys()
                    values = tuple(item[key] for key in keys)
                    cursor.execute('''
                        INSERT INTO {} ({}) VALUES ({})
                        '''.format(intent2table[category],', '.join(keys), ', '.join(['?']*len(values))), values)
                    all_items.append(item)       
        print(len(all_items))         
        all_items = remove_duplicate_dicts(all_items)
        print(len(all_items))
    
    conn.commit()
    for id,(originsample, outputsample) in enumerate(zip(origindata,outputdata)):
        # if id== 1:
            # exit()
        # newsample = deepcopy(originsample)
        scenario = originsample["scenario"]
        items = scenario["kb"]["items"]
        if dataset == "CamRest":
            items = remove_duplicates_by_id(items)   
        category = scenario["task"]["intent"]
        origindialogue = originsample["dialogue"]
        outputdialogue = outputsample["dialogue"]

        assert len(origindialogue) == len(outputdialogue)

        if items ==  None:
            new_samples.append(originsample)
            continue
        # for item in items:
        #     keys = item.keys()
        #     values = tuple(item[key] for key in keys)
        #     cursor.execute('''
        #         INSERT INTO {} ({}) VALUES ({})
        #         '''.format(intent2table[category],', '.join(keys), ', '.join(['?']*len(values))), values)
        # conn.commit()
        for utter_id, utterance in enumerate(origindialogue):
            
            if utterance["turn"] == "user":
                continue
            if utterance["turn"] == "system":
                # print(utter_id)
                output_utterance = outputdialogue[utter_id]
                annotated_query = utterance["annotated_query"]
                annotated_knowledge = utterance["annotated_knowledge"]
                del utterance["sql"]
                if "response" not in output_utterance:
                    # 说明模型生成的SQL格式可能有问题，无法提取知识
                    # 将annotated_knowledge 设置为空
                    utterance["response"] = ""
                    utterance["sql_extract"] = output_utterance["sql_extract"]
                    utterance["annotated_knowledge"] = []
                    
                else:
                    # print()
                    response = output_utterance["response"]
                    utterance["response"] = response
                    utterance["sql_extract"] = output_utterance["sql_extract"]
                    cursor.execute(response)
                    row = cursor.fetchall()
                    if dataset == "MultiWOZ":
                        sql_id_list = [t[0] for t in row]
                        # annotated_id_list = [d["name"] for d in annotated_knowledge if d["label"] == '1']
                        new_annotated_knowlegde =[]
                        for d in all_items:
                            if d["name"] in sql_id_list:
                                # d["label"] = "1"
                                knowledge = deepcopy(d)
                                knowledge["label"] ="1"

                            # else:
                            #     d["label"] = "0"
                                new_annotated_knowlegde.append(knowledge)
                        utterance["annotated_knowledge"] = new_annotated_knowlegde
                    elif dataset =="CamRest":
                        sql_id_list = [t[0] for t in row]
                        # print(row)
                        # annotated_id_list = [d["id"] for d in annotated_knowledge if d["label"] == '1']
                        new_annotated_knowlegde = []
                        for d in all_items:
                            if d["id"] in sql_id_list:
                                # d["label"] = "1"
                                knowledge = deepcopy(d)
                                knowledge["label"] ="1"
                            # else:
                            #     d["label"] = "0"
                                new_annotated_knowlegde.append(knowledge)
                        # print(new_annotated_knowlegde)
                        utterance["annotated_knowledge"] = new_annotated_knowlegde
                    else:
                        sql_id_list = [t[1:] for t in row]
                        sql_id_list = [tuple(sorted(t)) for t in sql_id_list]

                        # annotated_id_list = []
                        new_annotated_knowlegde =[]
                        for d in all_items:
                            tuple_of_values = tuple(v for k, v in d.items() if k != 'label')
                            tuple_of_values = sorted(tuple_of_values)
                            # if knowledge["label"] == "1":
                                # annotated_id_list.append(tuple(tuple_of_values))
                            if(tuple(tuple_of_values) in sql_id_list):
                                knowledge = deepcopy(d)
                                knowledge["label"] = "1"
                                new_annotated_knowlegde.append(knowledge)
                            # else:
                            #     knowledge["label"] = "0"
                            #     new_annotated_knowlegde.append(knowledge)
                        utterance["annotated_knowledge"] = new_annotated_knowlegde

        # cursor.execute('DELETE from {};'.format(intent2table[category]))     
        new_samples.append(originsample)          
        # os.makedirs("script/newtest/{}_output_{}".format(template_name,trainsql_version),exist_ok=True)
    with open(writedatapath,"w") as f:
        json.dump(new_samples,f)
        # cursor.execute("DROP TABLE  IF EXISTS RESTAURANT")
    if dataset=="MultiWOZ":
        cursor.execute("DROP TABLE  IF EXISTS RESTAURANT")          
        cursor.execute("DROP TABLE  IF EXISTS HOTEL")             
        cursor.execute("DROP TABLE  IF EXISTS ATTRACTION")       
    elif dataset == "CamRest":
        cursor.execute("DROP TABLE  IF EXISTS RESTAURANT")       
    # else:
    #     cursor.execute("DROP TABLE  IF EXISTS SCHEDULE")       
    #     cursor.execute("DROP TABLE  IF EXISTS NAVIGATE")       
    #     cursor.execute("DROP TABLE  IF EXISTS WEATHER")    
 

def extract_large_kbv2_by_sql(origindatapath, modeloutputpath, writedatapath, dataset):
    #! large 只针对MultiWOZ 和 CamRest
    with open(origindatapath,"r") as f:
        origindata = json.load(f)
    with open(modeloutputpath,"r") as f: 
        outputdata = json.load(f)
    assert len(origindata) == len(outputdata)
    intent2table = {
        "restaurant":"RESTAURANT",
        "schedule":"SCHEDULE",
        "attraction":"ATTRACTION",
        "hotel":"HOTEL",
        "weather":"WEATHER",
        "navigate":"NAVIGATE"
    }
    conn = sqlite3.connect(':memory:')
    cursor = conn.cursor()
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
    
    new_samples = []
    #! 构建大知识库
    all_items = []
    all_names = []
    for id, sample in enumerate(origindata):
        dialogue = sample["dialogue"]
        scenario = sample["scenario"]
        items = scenario["kb"]["items"]
        if dataset == "CamRest":
            items = remove_duplicates_by_id(items)    
        category = scenario["task"]["intent"]
        if dataset == "MultiWOZ":
            if category not in ["restaurant","attraction","hotel"]:
                continue
        
        for item in items:
            # if item not in all_items:
            if item["name"] not in all_names:
                # print(item["name"])
                keys = item.keys()
                values = tuple(item[key] for key in keys)
                cursor.execute('''
                    INSERT INTO {} ({}) VALUES ({})
                    '''.format(intent2table[category],', '.join(keys), ', '.join(['?']*len(values))), values)
                all_items.append(item)
                all_names.append(item["name"])
    assert len(all_items) == len(all_names)
    print(len(all_items))
        # all_items = remove_duplicate_dicts(all_items)
        # print(len(all_items))
        # exit()
    conn.commit()
    for id,(originsample, outputsample) in enumerate(zip(origindata,outputdata)):
        # if id== 1:
            # exit()
        # newsample = deepcopy(originsample)
        scenario = originsample["scenario"]
        items = scenario["kb"]["items"]
        if dataset == "CamRest":
            items = remove_duplicates_by_id(items)   
        category = scenario["task"]["intent"]
        origindialogue = originsample["dialogue"]
        outputdialogue = outputsample["dialogue"]

        assert len(origindialogue) == len(outputdialogue)

        if items ==  None:
            new_samples.append(originsample)
            continue
        # for item in items:
        #     keys = item.keys()
        #     values = tuple(item[key] for key in keys)
        #     cursor.execute('''
        #         INSERT INTO {} ({}) VALUES ({})
        #         '''.format(intent2table[category],', '.join(keys), ', '.join(['?']*len(values))), values)
        # conn.commit()
        for utter_id, utterance in enumerate(origindialogue):
            
            if utterance["turn"] == "user":
                continue
            if utterance["turn"] == "system":
                # print(utter_id)
                output_utterance = outputdialogue[utter_id]
                annotated_query = utterance["annotated_query"]
                annotated_knowledge = utterance["annotated_knowledge"]
                del utterance["sql"]
                if "response" not in output_utterance:
                    # 说明模型生成的SQL格式可能有问题，无法提取知识
                    # 将annotated_knowledge 设置为空
                    utterance["response"] = ""
                    utterance["sql_extract"] = output_utterance["sql_extract"]
                    utterance["annotated_knowledge"] = []
                    
                else:
                    # print()
                    response = output_utterance["response"]
                    utterance["response"] = response
                    utterance["sql_extract"] = output_utterance["sql_extract"]
                    cursor.execute(response)
                    row = cursor.fetchall()
                    if dataset == "MultiWOZ":
                        sql_id_list = [t[0] for t in row]
                        # annotated_id_list = [d["name"] for d in annotated_knowledge if d["label"] == '1']
                        new_annotated_knowlegde =[]
                        for d in all_items:
                            if d["name"] in sql_id_list:
                                # d["label"] = "1"
                                knowledge = deepcopy(d)
                                knowledge["label"] ="1"

                            # else:
                            #     d["label"] = "0"
                                new_annotated_knowlegde.append(knowledge)
                        utterance["annotated_knowledge"] = new_annotated_knowlegde[:7]
                    elif dataset =="CamRest":
                        sql_id_list = [t[0] for t in row]
                        # print(row)
                        # annotated_id_list = [d["id"] for d in annotated_knowledge if d["label"] == '1']
                        new_annotated_knowlegde = []
                        for d in all_items:
                            if d["id"] in sql_id_list:
                                # d["label"] = "1"
                                knowledge = deepcopy(d)
                                knowledge["label"] ="1"
                            # else:
                            #     d["label"] = "0"
                                new_annotated_knowlegde.append(knowledge)
                        # print(new_annotated_knowlegde)
                        utterance["annotated_knowledge"] = new_annotated_knowlegde[:7]
                    # else:
                    #     sql_id_list = [t[1:] for t in row]
                    #     sql_id_list = [tuple(sorted(t)) for t in sql_id_list]

                    #     # annotated_id_list = []
                    #     new_annotated_knowlegde =[]
                    #     for d in all_items:
                    #         tuple_of_values = tuple(v for k, v in d.items() if k != 'label')
                    #         tuple_of_values = sorted(tuple_of_values)
                    #         # if knowledge["label"] == "1":
                    #             # annotated_id_list.append(tuple(tuple_of_values))
                    #         if(tuple(tuple_of_values) in sql_id_list):
                    #             knowledge = deepcopy(d)
                    #             knowledge["label"] = "1"
                    #             new_annotated_knowlegde.append(knowledge)
                    #         # else:
                    #         #     knowledge["label"] = "0"
                    #         #     new_annotated_knowlegde.append(knowledge)
                    #     utterance["annotated_knowledge"] = new_annotated_knowlegde

        # cursor.execute('DELETE from {};'.format(intent2table[category]))     
        new_samples.append(originsample)          
        # os.makedirs("script/newtest/{}_output_{}".format(template_name,trainsql_version),exist_ok=True)
    with open(writedatapath,"w") as f:
        json.dump(new_samples,f)
        # cursor.execute("DROP TABLE  IF EXISTS RESTAURANT")
    if dataset=="MultiWOZ":
        cursor.execute("DROP TABLE  IF EXISTS RESTAURANT")          
        cursor.execute("DROP TABLE  IF EXISTS HOTEL")             
        cursor.execute("DROP TABLE  IF EXISTS ATTRACTION")       
    elif dataset == "CamRest":
        cursor.execute("DROP TABLE  IF EXISTS RESTAURANT")       
    # else:
    #     cursor.execute("DROP TABLE  IF EXISTS SCHEDULE")       
    #     cursor.execute("DROP TABLE  IF EXISTS NAVIGATE")       
    #     cursor.execute("DROP TABLE  IF EXISTS WEATHER")    
 

def obfuscate_sql(sql):
  """
  Obscures the SELECT statement in a provided SQL query by replacing everything between SELECT and FROM with asterisks.

  Args:
    sql: The SQL query string.

  Returns:
    The obfuscated SQL query string.
  """
  return re.sub(r"SELECT\s+(.*?)\s+FROM", "SELECT * FROM", sql, flags=re.IGNORECASE)
def extract_all_kb_by_sql(origindatapath, modeloutputpath, writedatapath, dataset):
    all_kb = json.load(open("dialogue/all_db.json", "r"))
    #! large 只针对MultiWOZ 和 CamRest
    with open(origindatapath,"r") as f:
        origindata = json.load(f)
    with open(modeloutputpath,"r") as f: 
        outputdata = json.load(f)
    assert len(origindata) == len(outputdata)
    intent2table = {
        "restaurant":"RESTAURANT",
        "schedule":"SCHEDULE",
        "attraction":"ATTRACTION",
        "hotel":"HOTEL",
        "weather":"WEATHER",
        "navigate":"NAVIGATE"
    }
    conn = sqlite3.connect(':memory:')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE MultiWOZ (
            name TEXT PRIMARY KEY,
            food TEXT,    
            address TEXT,
            area TEXT,
            domain TEXT,
            internet TEXT,
            parking TEXT,
            phone TEXT,
            postcode TEXT,
            pricerange TEXT,
            stars TEXT,
            type TEXT,
        )''')
    # if dataset == "MultiWOZ":
    #     cursor.execute('''
    #     CREATE TABLE RESTAURANT (
    #         name TEXT PRIMARY KEY,
    #         food TEXT,    
    #         address TEXT,
    #         area TEXT,
    #         phone TEXT,
    #         postcode TEXT,
    #         pricerange TEXT,
    #         stars TEXT,
    #         type TEXT,
    #         choice TEXT,
    #         ref TEXT
    #     )''')
    #     cursor.execute('''
    #     CREATE TABLE HOTEL (
    #     name TEXT PRIMARY KEY,
    #     address TEXT,
    #     area TEXT,
    #     phone TEXT,
    #     postcode TEXT,
    #     pricerange TEXT,
    #     stars TEXT,
    #     type TEXT,
    #     choice TEXT,
    #     ref TEXT,
    #     parking TEXT,
    #     interent TEXT
    # )''')
    
    #     cursor.execute('''
    #     CREATE TABLE ATTRACTION (
    #         name TEXT PRIMARY KEY,
    #         address TEXT,
    #         area TEXT,
    #         phone TEXT,
    #         postcode TEXT,
    #         pricerange TEXT,
    #         stars TEXT,
    #         type TEXT,
    #         choice TEXT,
    #         ref TEXT
    #     )''')
    # elif dataset =="CamRest":
    #     cursor.execute('''
    #     CREATE TABLE RESTAURANT (
    #         id TEXT PRIMARY KEY,
    #         name TEXT,
    #         area TEXT,
    #         food TEXT,
    #         phone TEXT,
    #         pricerange TEXT,
    #         location TEXT,
    #         address TEXT,
    #         type TEXT,
    #         postcode TEXT
    #     )''')
    
    new_samples = []
    #! 构建大知识库
    all_items = []
    all_names = []
    for id, sample in enumerate(origindata):
        dialogue = sample["dialogue"]
        scenario = sample["scenario"]
        items = scenario["kb"]["items"]
        if dataset == "CamRest":
            items = remove_duplicates_by_id(items)    
        category = scenario["task"]["intent"]
        if dataset == "MultiWOZ":
            if category not in ["restaurant","attraction","hotel"]:
                continue
        
        # for item in items:
        #     # if item not in all_items:
        #     if item["name"] not in all_names:
        #         # print(item["name"])
        #         keys = item.keys()
        #         values = tuple(item[key] for key in keys)
        #         cursor.execute('''
        #             INSERT INTO {} ({}) VALUES ({})
        #             '''.format(intent2table[category],', '.join(keys), ', '.join(['?']*len(values))), values)
        #         all_items.append(item)
        #         all_names.append(item["name"])
    assert len(all_items) == len(all_names)
    print(len(all_items))
        # all_items = remove_duplicate_dicts(all_items)
        # print(len(all_items))
        # exit()
    conn.commit()
    for id,(originsample, outputsample) in enumerate(zip(origindata,outputdata)):
        # if id== 1:
            # exit()
        # newsample = deepcopy(originsample)
        scenario = originsample["scenario"]
        items = scenario["kb"]["items"]
        if dataset == "CamRest":
            items = remove_duplicates_by_id(items)   
        category = scenario["task"]["intent"]
        origindialogue = originsample["dialogue"]
        outputdialogue = outputsample["dialogue"]

        assert len(origindialogue) == len(outputdialogue)

        if items ==  None:
            new_samples.append(originsample)
            continue
        # for item in items:
        #     keys = item.keys()
        #     values = tuple(item[key] for key in keys)
        #     cursor.execute('''
        #         INSERT INTO {} ({}) VALUES ({})
        #         '''.format(intent2table[category],', '.join(keys), ', '.join(['?']*len(values))), values)
        # conn.commit()
        for utter_id, utterance in enumerate(origindialogue):
            
            if utterance["turn"] == "user":
                continue
            if utterance["turn"] == "system":
                # print(utter_id)
                output_utterance = outputdialogue[utter_id]
                annotated_query = utterance["annotated_query"]
                annotated_knowledge = utterance["annotated_knowledge"]
                del utterance["sql"]
                if "response" not in output_utterance:
                    # 说明模型生成的SQL格式可能有问题，无法提取知识
                    # 将annotated_knowledge 设置为空
                    utterance["response"] = ""
                    utterance["sql_extract"] = output_utterance["sql_extract"]
                    utterance["annotated_knowledge"] = []
                    
                else:
                    # print()
                    response = output_utterance["response"]
                    utterance["response"] = response
                    utterance["sql_extract"] = output_utterance["sql_extract"]
                    cursor.execute(response)
                    row = cursor.fetchall()
                    if dataset == "MultiWOZ":
                        sql_id_list = [t[0] for t in row]
                        # annotated_id_list = [d["name"] for d in annotated_knowledge if d["label"] == '1']
                        new_annotated_knowlegde =[]
                        for d in all_items:
                            if d["name"] in sql_id_list:
                                # d["label"] = "1"
                                knowledge = deepcopy(d)
                                knowledge["label"] ="1"

                            # else:
                            #     d["label"] = "0"
                                new_annotated_knowlegde.append(knowledge)
                        utterance["annotated_knowledge"] = new_annotated_knowlegde
                    elif dataset =="CamRest":
                        sql_id_list = [t[0] for t in row]
                        # print(row)
                        # annotated_id_list = [d["id"] for d in annotated_knowledge if d["label"] == '1']
                        new_annotated_knowlegde = []
                        for d in all_items:
                            if d["id"] in sql_id_list:
                                # d["label"] = "1"
                                knowledge = deepcopy(d)
                                knowledge["label"] ="1"
                            # else:
                            #     d["label"] = "0"
                                new_annotated_knowlegde.append(knowledge)
                        # print(new_annotated_knowlegde)
                        utterance["annotated_knowledge"] = new_annotated_knowlegde
                    # else:
                    #     sql_id_list = [t[1:] for t in row]
                    #     sql_id_list = [tuple(sorted(t)) for t in sql_id_list]

                    #     # annotated_id_list = []
                    #     new_annotated_knowlegde =[]
                    #     for d in all_items:
                    #         tuple_of_values = tuple(v for k, v in d.items() if k != 'label')
                    #         tuple_of_values = sorted(tuple_of_values)
                    #         # if knowledge["label"] == "1":
                    #             # annotated_id_list.append(tuple(tuple_of_values))
                    #         if(tuple(tuple_of_values) in sql_id_list):
                    #             knowledge = deepcopy(d)
                    #             knowledge["label"] = "1"
                    #             new_annotated_knowlegde.append(knowledge)
                    #         # else:
                    #         #     knowledge["label"] = "0"
                    #         #     new_annotated_knowlegde.append(knowledge)
                    #     utterance["annotated_knowledge"] = new_annotated_knowlegde

        # cursor.execute('DELETE from {};'.format(intent2table[category]))     
        new_samples.append(originsample)          
        # os.makedirs("script/newtest/{}_output_{}".format(template_name,trainsql_version),exist_ok=True)
    with open(writedatapath,"w") as f:
        json.dump(new_samples,f)
        # cursor.execute("DROP TABLE  IF EXISTS RESTAURANT")
    if dataset=="MultiWOZ":
        cursor.execute("DROP TABLE  IF EXISTS RESTAURANT")          
        cursor.execute("DROP TABLE  IF EXISTS HOTEL")             
        cursor.execute("DROP TABLE  IF EXISTS ATTRACTION")       
    elif dataset == "CamRest":
        cursor.execute("DROP TABLE  IF EXISTS RESTAURANT")       
    # else:
    #     cursor.execute("DROP TABLE  IF EXISTS SCHEDULE")       
    #     cursor.execute("DROP TABLE  IF EXISTS NAVIGATE")       
    #     cursor.execute("DROP TABLE  IF EXISTS WEATHER")    
 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kb_version", type=str, default='', help="")
    parser.add_argument("--model_version", type=str,default="")
    parser.add_argument("--zero_shot", action="store_true",)
    parser.add_argument("--test_intent", type=str, default="")
    args = parser.parse_args()
    kb_version = args.kb_version
    model_version = args.model_version
    zero_shot = args.zero_shot
    intent = args.test_intent
    modes = ["test"]
    models = ["qwen"]
    
    if zero_shot:
        for mode in modes:
            for model in models:
                origindatapath = "origindata/SMD_zero-shot/{}_test.json".format(intent[:3])
                outputdatapath = "script/newtest/{}_output_{}/{}_{}_{}.json".format(model,model_version,"SMD",mode,intent[:3])
                os.makedirs("dialogue/generate_data/small_kb-{}/{}".format(kb_version,model),exist_ok=True)
                write_datapath = "dialogue/generate_data/small_kb-{}/{}/{}_{}_{}.json".format(kb_version,model,"SMD",mode,intent[:3])
                extract_small_kb_by_sql(origindatapath,outputdatapath,write_datapath,"SMD")
    else:
        modes = ["test"]
        datasets = ["CamRest", "MultiWOZ", "SMD"]
        models = ["qwen",]
        for dataset in datasets:
            for mode in modes:
                for model in models:
                    origindatapath = "origindata/{}/sql/{}.json".format(dataset,mode)
                    outputdatapath = "script/newtest/{}_output_{}/{}_{}.json".format(model,model_version,dataset,mode)
                    os.makedirs("dialogue/generate_data/small_kb-{}/{}".format(kb_version,model),exist_ok=True)
                    write_datapath = "dialogue/generate_data/small_kb-{}/{}/{}_{}.json".format(kb_version,model,dataset,mode)

                    extract_small_kb_by_sql(origindatapath,outputdatapath,write_datapath,dataset)

        # ! large base

        # modes = ["train","test"]
        # datasets = ["SMD"]
        # models = ["qwen","llama2"]
        # for dataset in datasets:
        #     for mode in modes:
        #         for model in models:
        #             origindatapath = "origindata/{}/sql/{}.json".format(dataset,mode)
        #             outputdatapath = "script/newtest/{}_output_sftdataexactsqlv3-2/{}_{}.json".format(model,dataset,mode)
        #             os.makedirs("dialogue/generate_data/large_kb/{}".format(model),exist_ok=True)
        #             write_datapath = "dialogue/generate_data/large_kb/{}/{}_{}.json".format(model,dataset,mode)

        #             extract_large_kb_by_sql(origindatapath,outputdatapath,write_datapath,dataset)


        # modes = ["train","test"]
        # datasets = ["CamRest", "MultiWOZ"]
        # models = ["qwen","llama2"]
        # for dataset in datasets:
        #     for mode in modes:
        #         for model in models:
        #             origindatapath = "origindata/{}/sql/{}.json".format(dataset,mode)
        #             outputdatapath = "script/newtest/{}_output_sql3-2-dial-50/{}_{}.json".format(model,dataset,mode)
        #             os.makedirs("dialogue/generate_data/large_kbv2/{}".format(model),exist_ok=True)
        #             write_datapath = "dialogue/generate_data/large_kbv2/{}/{}_{}.json".format(model,dataset,mode)

        #             extract_large_kbv2_by_sql(origindatapath,outputdatapath,write_datapath,dataset)



    
