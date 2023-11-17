import json
from VIC import check
import openai
import torch
from test_api import check_api
import sys
sys.path.append("../LAVIS/lavis/")
from models import load_model_and_preprocess
from PIL import Image
import openai
from pipeline import get_edits
import time

### Change the file paths
caps_lst_path = "Random_cap.json"
guidance_lst_path = '../Guidance.json'
taxonomy_path = '../Index2G.json'
API_key_lst_path = "../Week16/key_5.txt"

### Read the files
with open(caps_lst_path, 'r') as f:
    caps_lst = json.load(f)
with open(guidance_lst_path, 'r') as f:
    index_lst = json.load(f)
with open(taxonomy_path, 'r') as f:
    obj_dict = json.load(f)
with open(API_key_lst_path, "r") as f:
    content = f.read()
key_lst = content.split("\n")
openai.api_key = key_lst[0]

def get_index(index, caps_lst):
    for i in range(len(caps_lst)):
        if caps_lst[i]["index"] == index:
            return i
        
def get_guidance(index):
    return index_lst[str(index)]

temp = []
res_cap_lst = []
error_lst = []
start_time = time.time()

for subset_i in range(len(caps_lst)):
    index = caps_lst[subset_i]["index"]
    if index in list(index_lst.keys()):
        try:
            guidance = get_guidance(index)
            guidance_lst = guidance.split("|")
            objective_lst = []
            for g in guidance_lst:
                pre = g.split("-")[0]
                con = g.split("-")[1]
                objective = obj_dict[pre][con]
                objective_lst.append(objective.strip())
            objective = " and ".join(objective_lst)
            print(index)
            print(subset_i)
            print(objective)
            cap_dict = {}
            cap = caps_lst[get_index(index, caps_lst)]
            Edits, q_lst, a_s_lst, a_e_lst, log, key_lst = get_edits(index = index, key_lst = key_lst, subject = objective, device = torch.device("cuda:0"))
            cap_dict["index"] = cap["index"]
            cap_dict["Edits"] = Edits
            cap_dict["sup_q_lst"] =q_lst
            cap_dict["sup_a_s_lst"] = a_s_lst
            cap_dict["sup_a_e_lst"] = a_e_lst
            cap_dict["Log"] = log
            res_cap_lst.append(cap_dict)
        except:
            error_lst.append(index)
            for key in key_lst:
                if check_api(key) == False:
                    key_lst.remove(key)
            if len(key_lst) == 0:
                print("No Valid Key!!!!!")
                print("Stop from the cap_i: {}".format(index))
                print("Stop from the index: {}".format(index))
                with open("Eval/Guided_Edits.json", "w") as f:
                    json.dump(res_cap_lst, f, indent = 4)
                end_time = time.time()
                exe_time = end_time - start_time
                num_caps = len(res_cap_lst)
                with open("Eval/time_cost.txt", "w") as file:
                    content = "Total time: {}\nTotal Num: {}\nAvg. time cost: {}".format(exe_time, num_caps, float(exe_time / num_caps))
                    file.write(content)
                with open('Eval/error_Guided_Edits.txt', 'w') as file:
                    for item in error_lst:
                        file.write(item + '\n')
                sys.exit(0)
end_time = time.time()
exe_time = end_time - start_time
num_caps = len(res_cap_lst)

with open("Eval/time_cost.txt", "w") as file:
    content = "Total time: {}\nTotal Num: {}\nAvg. time cost: {}".format(exe_time, num_caps, float(exe_time / num_caps))
    file.write(content)
    
with open("Eval/Guided_Edits.json", "w") as f:
    json.dump(res_cap_lst, f, indent = 4)
with open('Eval/error_Guided_Edits.txt', 'w') as file:
    for item in error_lst:
        file.write(item + '\n')