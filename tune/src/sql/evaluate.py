import json
import argparse
def caculate_sql_extract_origindata(filepath):
    with open(filepath,"r") as f:
        data = json.load(f)
        all = 0
        right = 0
        exact_right = 0
        wrong = 0
        for sample in data:
            dialogue = sample["dialogue"]
            items = sample["scenario"]["kb"]["items"]
            if items==None:
                continue
            for utterance in dialogue:
                if utterance["turn"] == "user":
                    continue
                if utterance["turn"] == "system":
                    all += 1
                    # print(utterance["utterance"])
                    sql_extract = utterance["sql_extract"]
                    if sql_extract == "exact_right":
                        exact_right += 1
                        right += 1
                    elif sql_extract == "right":
                        right += 1
                    else:
                        wrong += 1

    print("### ",filepath)
    print(all)
    print(right) 
    print(exact_right)
    print(wrong)
    print(right/all)
    print(exact_right/all)
    print("\n\n")
    return all, right, exact_right, wrong, right/all, exact_right/all


def caculate_sql_extract_exactdata(filepath):
    with open(filepath,"r") as f:
        all = 0
        right = 0
        exact_right = 0
        wrong = 0
        for line in f.readlines():
            all += 1
            line = json.loads(line)
            sql_extract = line["response_extract"]
            if sql_extract == "exact_right":
                exact_right += 1
                right += 1
            elif sql_extract == "right":
                right += 1
            else:
                wrong += 1
    print("==============")
    print(filepath)
    print(all)
    print(right) 
    print(exact_right)
    print(wrong)
    print(right/all)
    print(exact_right/all)
    print("==============\n")
    return all, right, exact_right, wrong, right/all, exact_right/all


if __name__== "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", type=str)
    args = parser.parse_args()
    json_path = args.json_path
    # modes = ["train","test"]
    # datasets = ["CamRest", "MultiWOZ", "SMD"]
    # sql_version = "sql3-2-dial-50"
    # for dataset in datasets:
    #     for mode in modes:

    #         # filepath = "origindata/new_updatesql/{}/{}_{}.json".format(sql_version,dataset,mode)
    #         filepath = "script/newtest/qwen_output_{}/{}_{}.json".format(sql_version,dataset,mode)
    #         caculate_sql_extract_origindata(filepath)
    # caculate_sql_extract_origindata("script/newtest/qwen_output_sql3-2-dial-50/CamRest_train.json")
    # caculate_sql_extract_origindata("script/newtest/qwen_output_exactsqlv2/MultiWOZ_train.json")
    # caculate_sql_extract_origindata("script/newtest/qwen_output_zero-shot/SMD_test_wea.json")
    # caculate_sql_extract_origindata("script/newtest/qwen_output_CS-sql/MultiWOZ_test.json")
    # caculate_sql_extract_origindata("script/newtest/qwen_output_MS-sql/CamRest_test.json")
    caculate_sql_extract_origindata(json_path)


# 
# script/newtest/qwen_output_sql3-2-dial-50
    # caculate_sql_extract_exactdata("glm_res/sql_gpt/SMD_20.jsonl")


