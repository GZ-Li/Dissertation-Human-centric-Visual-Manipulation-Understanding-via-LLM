import json
import torchvision
import torch
import jsonlines
import requests
import pandas as pd
import os
import torch
from PIL import Image
import sys
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
def get_subject_mental(edits_lst, cap, subject = "subject1", attr = "Edits"):

    men_prompt = '''
Help me to infer the Edits' effects on the subjects in the image? Mental states refer to how the edits can mislead people into thinking something about the subjects that is actually not true. At the end of this prompt, it will remind you which subject to infer, such as 'Subject1', 'Subject2', etc. The Edits are applied to the Source Image. If a famous person is the subject of the edit, then the focus should be on the implications related to that person, such as making fun of them. Besides, pay attention to any object exchanges within the image, like face or clothes swaps. Also, try to substitute the characters mentioned in your answer as 'Subject1', 'Subject2', and so on, based on the 'Subject' defined in the 'Question:'. The source image description is written after 'Source Image:', and the edited image description is written after 'Edited Image:'. According to the question after 'Question:', write your answer after 'Subject:'.

Edits:
The man in the source image is replaced with a man wearing sunglasses and riding an ostrich in the target image.
The man in the source image is looking down with his head in the other hand, while in the target image, the man is smiling with his eyes closed.
Source Image:
The image depicts a man hanging from a high building with his feet on a rope and holding a cat in one hand and looking down with his head in the other.
Question:
What is the mental state of Subject1? Subject1 is main characters in both the Source Image and the Edited Image. In the Edited Image, subject1 is a man sitting on an ostrich with sunglasses on his head and smiling at the camera.
Subject1:
Subject1 would find it hilarious, a funny photo of him at work was used to make it look like he's riding an ostrich.

Edits: 
The astronaut walking on the moon with a stick and poles in his hand is replaced by a zombie walking on the moon with a man in a suit and tie.
The vehicle in the background is replaced by a dog with a bloody face and body.
The sky is replaced by a zombie climbing on a rock with a radio.
The man walking behind them is replaced by a man with blood on his face and a suit jacket.
Source Image:
The image depicts an astronaut walking on the moon with a stick and poles in his hand, and a vehicle in the background. The sky is dark and distant.
Question:
What is the mental state of Subject5? Subject5 is the main characters in both the Source Image and the Edited Image. In the Edited Image, subject5 is a man standing on the moon with his arms out and a knife in his hand.
Subject5:
Subject5 would feel breathless, he is on the moon where there is not enough oxygen for humans, even zombie humans.

Edits: 
The woman and the cat in the source image are replaced by a woman with a hat and a man with a sword in the target image.
Source Image:
The image depicts a woman and a cat playing in a yard with a fence behind them, a red fence in the background, bushes, and green grass.
Question:
What is the mental state of Subject3? Subject3 is the main character in both the Source Image and the Edited Image. In the Edited Image, subject3 is an older woman fighting with another man.
Subject3:
Subject3 would feel annoyed, his serious protest action is portrayed as flinging a video game character.

Edits:
The part in the source image a woman holding a bouquet of flowers and the other woman's hand up with her fingers is replaced by the part in the target image a woman holding a plate with her hands.
The part in the source image a woman giving flowers to another woman's hand is replaced by the part in the target image  a person holding a sign that says president of the united states.
The part in the target image a man with his head tilted is introduced.
Source Image:
The image depicts a group of women clapping and having fun with each other, with one woman holding a bouquet of flowers and the other woman's hand up with her fingers.
Question:
What is the mental state of Subject4? Subject4 is the main characters in both the Source Image and the Edited Image. In the Edited Image, subject4 is the woman holding a plaque with the seal of the president of the united states.
Subject4:
Subject4 would feel happy, it's being implied that she has become president of the United States she is seen celebrating with the presidential seal.

'''

    men_prompt = men_prompt + "Edits:"
    index = cap["index"]
    edits = extract_edits(edits_lst, index, attr = attr)
    for e in edits:
        men_prompt = men_prompt + "\n" + e
    men_prompt = men_prompt + "\n" + "Source Image:" + "\n" + cap["global_s"].strip() + " "
    # men_prompt = men_prompt + subject.capitalize() + " is the main character in both the Source Image and the Edited Image. "
    
    men_prompt = men_prompt + "\nQuestion:\n" + "What is the mental state of " + subject.capitalize() + "? "
    
    sub_lst = []
    for s in cap["subject_e"]:
        if s.startswith("subject"):
            sub_lst.append(s.split()[0])
    sub_lst = list(set(sub_lst))

    if subject in sub_lst:
        for s in cap["subject_e"]:
            if s.startswith(subject.lower()):
                men_prompt = men_prompt + "In the Edited Image, " + s

    men_prompt = men_prompt + "\n" + subject.capitalize() + ":\n"
    # men_prompt = men_prompt + "\nMental States:\n"

    # print(men_prompt)
    
    mental_states = openai.Completion.create(
                                engine="text-davinci-003",
                                prompt=men_prompt,
                                max_tokens=512,
                                n=1,
                                stop=None,
                                temperature=0,
                            )

    men_result = mental_states.choices[0].text
    
    return men_result.strip().strip("\n")

# TEST
# import openai
# openai.api_key = "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"
# with open('../Week22_3/Random_cap.json', 'r') as f:
#     data = json.load(f)
# print(get_subject_mental(data[20], attr = "Edits"))