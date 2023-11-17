import json
import torchvision
import torch
import jsonlines
import requests
import pandas as pd
import os
import torch
from PIL import Image
# import sys
# sys.path.append("../Lavis/LAVIS/lavis/")
# from models import load_model_and_preprocess
#from lavis.models import load_model_and_preprocess
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import openai
import time
# from test_api import check_api
import random

def extract_edits(edits_lst, index, attr = "Edits"):
    for i in range(len(edits_lst)):
        if edits_lst[i]["index"] == index:
            if edits_lst[i][attr] == "No Difference!":
                temp_lst = []
            else:
                temp_lst = edits_lst[i][attr]
            return temp_lst

# INPUT: A dictionary
def get_intent(edits_lst, cap, attr = "Edits"):

    int_prompt = '''
Help me to infer the intentions behind edits? Intention refers to the motivations of the editors and their intent to deceive in this context. The Edits are applied to the Source Image. If a famous person is the subject of the edit, then the focus should be on the implications related to that person, such as making fun of them. Besides, pay attention to any object exchanges within the image, like face or clothes swaps. The source image description is written after 'Source Image:', and the edits are written after 'Edits:'. Write your answer after 'Intents'.

Edits:
The man in the source image is replaced with a man wearing sunglasses and riding an ostrich in the target image.
The man in the source image is looking down with his head in the other hand, while in the target image, the man is smiling with his eyes closed.
Source Image:
The image depicts a man hanging from a high building with his feet on a rope and holding a cat in one hand and looking down with his head in the other.
Intents: 
Editor created this edit to utilize the unique pose the man was in, it was used to make it look like he's riding an ostrich while holding it's long neck.

Edits: 
The astronaut walking on the moon with a stick and poles in his hand is replaced by a zombie walking on the moon with a man in a suit and tie.
The vehicle in the background is replaced by a dog with a bloody face and body.
The sky is replaced by a zombie climbing on a rock with a radio.
The man walking behind them is replaced by a man with blood on his face and a suit jacket.
Source Image:
The image depicts an astronaut walking on the moon with a stick and poles in his hand, and a vehicle in the background. The sky is dark and distant.
Intents: 
Editor created this edit to make people laugh and scared by imagining zombies on the moon.

Edits: 
The woman and the cat in the source image are replaced by a woman with a hat and a man with a sword in the target image.
Source Image:
The image depicts a woman and a cat playing in a yard with a fence behind them, a red fence in the background, bushes, and green grass.
Intents: 
Editor created this edit to make the woman appear strong and dangerous, it looks like she is attacking another woman with a sword.

Edits:
The part in the source image a woman holding a bouquet of flowers and the other woman's hand up with her fingers is replaced by the part in the target image a woman holding a plate with her hands.
The part in the source image a woman giving flowers to another woman's hand is replaced by the part in the target image  a person holding a sign that says president of the united states.
The part in the target image a man with his head tilted is introduced.
Source Image:
The image depicts a group of women clapping and having fun with each other, with one woman holding a bouquet of flowers and the other woman's hand up with her fingers.
Intents:
Editor created this edit to make it look like the woman has become president of the USA, she is seen celebrating with other people while holding the presidential seal.

'''

    int_prompt = int_prompt + "Edits:"
    index = cap["index"]
    edits = extract_edits(edits_lst, index, attr = attr)
    for e in edits:
        int_prompt = int_prompt + "\n" + e
    int_prompt = int_prompt + "\n" + "Source Image:" + "\n" + cap["global_s"].strip() + " "
    
    # intro_sen = ""
    # sub_lst = []
    # for s in cap["subject_e"]:
    #     if s.startswith("subject"):
    #         sub_lst.append(s.split()[0])
    # sub_lst = list(set(sub_lst))

    # if len(cap["subject_e"]) > 0:
    #     if len(cap["subject_e"]) == 1:
    #         intro_sen = intro_sen + sub_lst[0] + " is the main character in both the Source Image and the Edited Image."
    #         intro_sen = intro_sen.capitalize()
    #     else:
    #         intro_sen = intro_sen + sub_lst[0]
    #         for sub in range(len(sub_lst) - 1):
    #             intro_sen = intro_sen + ", " + sub_lst[sub + 1]
    #         intro_sen = intro_sen + " are the main characters in both the Source Image and the Edited Image."
    #         intro_sen = intro_sen.capitalize()
    # int_prompt = int_prompt + intro_sen

    # if len(cap["subject_e"]) > 0:
    #     int_prompt = int_prompt + " In the Edited image"
    #     for s in cap["subject_e"]:
    #         int_prompt = int_prompt + ", " + s.strip(".")
    #     int_prompt = int_prompt + "."
    # else:
    #     int_prompt = int_prompt + " Subject1 is the main character in the image."

    int_prompt = int_prompt + "\nIntent:\n"

    # print(int_prompt)
    
    intents = openai.Completion.create(
                                engine="text-davinci-003",
                                prompt=int_prompt,
                                max_tokens=512,
                                n=1,
                                stop=None,
                                temperature=0,
                            )

    int_result = intents.choices[0].text
    
    return int_result.strip().strip("\n")

# TEST
# import openai
# openai.api_key = "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"
# with open('../Week22_3/Random_cap.json', 'r') as f:
#     data = json.load(f)
# print(get_intent(data[0], attr = "Checked Edits"))