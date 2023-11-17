from implication2 import get_implication
from intent2 import get_intent
from mental2 import get_subject_mental
from effect2 import get_subject_effect
from test_api import check_api
import openai
import json
import jsonlines
import time
import pandas as pd
import sys


caps_lst_path = "Random_cap.json"
API_key_lst_path = "../Week16/key_5.txt"
Dataset_path = "../EMU_Dataset/emu_dev.jsonl"


with open("../" + caps_lst_path, 'r') as f:
    checked_caps = json.load(f)
with open('Guided_Edits.json', 'r') as f:
    edits_lst = json.load(f)
with open("../" + API_key_lst_path, "r") as f:
    content = f.read()
key_lst_o = content.split("\n")
key_lst = key_lst_o
# key = key_lst[0]
# openai.api_key = key

def extract_cap(index, checked_caps):
    for i in range(len(checked_caps)):
        if checked_caps[i]["index"] == index:
            return checked_caps[i]

with open(Dataset_path, "r+", encoding="utf8") as f:
    source_lst = []
    edited_lst = []
    frame_lst = []
    response_lst = []
    rationale_lst = []
    label_lst = []
    index_lst = []
    box_lst = []
    error_lst = []
    for item in jsonlines.Reader(f):
        source_lst.append(item["source"])
        edited_lst.append(item["edited"])
        frame_lst.append(item["frame"])
        response_lst.append(item["response"])
        rationale_lst.append(item["rationale"])
        label_lst.append(item["label"])
        if item["metadata"] == {}:
            index_lst.append("")
            box_lst.append([])
        else:
            index_lst.append(item["metadata"]["index"])
            box_lst.append(item["metadata"]["bboxes"])

dev_data = pd.DataFrame()
dev_data["source"] = source_lst
dev_data["edited"] = edited_lst
dev_data["frame"] = frame_lst
dev_data["response"] = response_lst
dev_data["rationale"] = rationale_lst
dev_data["label"] = label_lst
dev_data["index"] = index_lst
dev_data["box"] = box_lst

checked_imp_error_lst = []
checked_imp_lst = []

for checked_cap_i in range(len(edits_lst)):
# for checked_cap_i in range(2):

    print("Implications")
    print(checked_cap_i)
    index = edits_lst[checked_cap_i]["index"]
    cap = extract_cap(index, checked_caps)

    try:

        key_lst = key_lst[1:] + [key_lst[0]]
        key = key_lst[0]
        openai.api_key = key
        implication = get_implication(edits_lst, cap, attr = "Edits")
        cap["checked_implications"] = implication
    
    except:
        
        checked_imp_error_lst.append(cap["index"])
        if check_api(key) == False:
            key_lst.remove(key)
        if len(key_lst) == 0:
            print("No valid key!!!!!!")
            sys.exit(0)
            time.sleep(100)

    checked_imp_lst.append(cap)

with open("pipeline_imp_cap.json", "w") as f:
    json.dump(checked_imp_lst, f, indent = 4)
with open('error_pipeline_imp_cap.txt', 'w') as file:
    for item in checked_imp_error_lst:
        file.write(item + '\n')


# ##############################################################


checked_int_error_lst = []
checked_int_lst = []

for checked_cap_i in range(len(edits_lst)):
# for checked_cap_i in range(2):

    print("Intents")
    print(checked_cap_i)
    index = edits_lst[checked_cap_i]["index"]
    cap = extract_cap(index, checked_caps)

    try:

        key_lst = key_lst[1:] + [key_lst[0]]
        key = key_lst[0]
        openai.api_key = key
        intent = get_intent(edits_lst, cap, attr = "Edits")
        cap["checked_intents"] = intent
    
    except:
        
        checked_int_error_lst.append(cap["index"])
        if check_api(key) == False:
            key_lst.remove(key)
        if len(key_lst) == 0:
            print("No valid key!!!!!!")
            sys.exit(0)
            time.sleep(100)

    checked_int_lst.append(cap)

with open("pipeline_int_cap.json", "w") as f:
    json.dump(checked_int_lst, f, indent = 4)
with open('error_pipeline_int_cap.txt', 'w') as file:
    for item in checked_int_error_lst:
        file.write(item + '\n')

##################################################################


checked_men_error_lst = []
checked_men_lst = []

for checked_cap_i in range(len(edits_lst)):
# for checked_cap_i in range(2):

    print("Mental States")
    print(checked_cap_i)

    subject_mental_dict = {}
    index = edits_lst[checked_cap_i]["index"]
    cap = extract_cap(index, checked_caps)
    temp_dev = dev_data[dev_data["index"] == index]

    frame_lst = list(set(temp_dev["frame"]))
    sub_mental_lst = []
    for f in frame_lst:
        if (f.startswith("subject")) and (len(f.split("_")) == 3):
            sub_mental_lst.append(f)
    sub_mental_lst = list(set(sub_mental_lst))
    for s in sub_mental_lst:
        try:
            key_lst = key_lst[1: ] + [key_lst[0]]
            key = key_lst[0]
            openai.api_key = key
            mental = get_subject_mental(edits_lst, cap, subject = str(s.split("_")[0]), attr = "Edits")
            subject_mental_dict[s] = mental
        except:
            checked_men_error_lst.append(cap["index"])
            if check_api(key) == False:
                key_lst.remove(key)  
            if len(key_lst) == 0:
                print("No valid keys!!!")
                sys.exit(0)
                time.sleep(100)  
    cap["subjects_mental_states"] = subject_mental_dict
    checked_men_lst.append(cap)

with open("pipeline_men_cap.json", "w") as f:
    json.dump(checked_men_lst, f, indent = 4)
with open('error_pipeline_men_cap.txt', 'w') as file:
    for item in checked_men_error_lst:
        file.write(item + '\n')


# # ###################################################################


checked_eff_error_lst = []
checked_eff_lst = []

for checked_cap_i in range(len(edits_lst)):
# for checked_cap_i in range(2):

    print("Effects")
    print(checked_cap_i)

    subject_effect_dict = {}
    index = edits_lst[checked_cap_i]["index"]
    cap = extract_cap(index, checked_caps)
    temp_dev = dev_data[dev_data["index"] == index]

    frame_lst = list(set(temp_dev["frame"]))
    sub_effect_lst = []
    for f in frame_lst:
        if (f.startswith("subject")) and (len(f.split("_")) == 2):
            sub_effect_lst.append(f)
    sub_effect_lst = list(set(sub_effect_lst))
    for s in sub_effect_lst:
        try:
            key_lst = key_lst[1: ] + [key_lst[0]]
            key = key_lst[0]
            openai.api_key = key
            mental = get_subject_effect(edits_lst, cap, subject = str(s.split("_")[0]), attr = "Edits")
            subject_effect_dict[s] = mental
        except:
            checked_eff_error_lst.append(cap["index"])
            if check_api(key) == False:
                key_lst.remove(key)  
            if len(key_lst) == 0:
                print("No valid keys!!!")
                sys.exit(0)
                time.sleep(100)  
    cap["subjects_effects"] = subject_effect_dict
    checked_eff_lst.append(cap)

with open("pipeline_eff_cap.json", "w") as f:
    json.dump(checked_eff_lst, f, indent = 4)
with open('error_pipeline_eff_cap.txt', 'w') as file:
    for item in checked_eff_error_lst:
        file.write(item + '\n')