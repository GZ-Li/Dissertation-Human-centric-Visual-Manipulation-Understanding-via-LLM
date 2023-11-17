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
def get_subject_effect(edits_lst, cap, subject = "subject1", attr = "Edits"):

    eff_prompt = '''
Help me to infer the Edits' effects on the subjects in the image? Effects refer to how the edits can mislead people into thinking something about the subjects that is actually not true. At the end of this prompt, it will remind you which subject to infer, such as 'Subject1', 'Subject2', etc. The Edits are applied to the Source Image. If a famous person is the subject of the edit, then the focus should be on the implications related to that person, such as making fun of them. Besides, pay attention to any object exchanges within the image, like face or clothes swaps. Also, try to substitute the characters mentioned in your answer as 'Subject1', 'Subject2', and so on, based on the 'Subject' defined in the 'Question:'. The source image description is written after 'Source Image:', and the edited image description is written after 'Edited Image:'. According to the question after 'Question:', write your answer after 'Subject:'.

Edits: 
The astronaut walking on the moon with a stick and poles in his hand is replaced by a zombie walking on the moon with a man in a suit and tie.
The vehicle in the background is replaced by a dog with a bloody face and body.
The sky is replaced by a zombie climbing on a rock with a radio.
The man walking behind them is replaced by a man with blood on his face and a suit jacket.
Source Image:
The image depicts an astronaut walking on the moon with a stick and poles in his hand, and a vehicle in the background. The sky is dark and distant.
Question:
What are the effects of the edits? Subject1 is the main characters in both the Source Image and the Edited Image. In the Edited Image, subject1 is a man standing on the moon with his arms out and a knife in his hand.
Subject1:
Someone might mistakenly believe that Subject1 is an astronaut who died on the moon and became a zombie, zombies can't fly rockets.

Edits:
The part in the source image a woman holding a bouquet of flowers and the other woman's hand up with her fingers is replaced by the part in the target image a woman holding a plate with her hands.
The part in the source image a woman giving flowers to another woman's hand is replaced by the part in the target image  a person holding a sign that says president of the united states.
The part in the target image a man with his head tilted is introduced.
Source Image:
The image depicts a group of women clapping and having fun with each other, with one woman holding a bouquet of flowers and the other woman's hand up with her fingers.
Question:
What are the effects of the edits? Subject4 is the main character in both the Source Image and the Edited Image. In the Edited Image, subject4 is the woman holding a plaque with the seal of the president of the united states.
Subject4:
Someone might mistakenly believe that Subject1 has become president of the USA, she is seen celebrating with other people while holding the presidential seal.

Edits:
The man in the fur coat is replaced by a group of people dressed in medieval clothing.
Source Image:
The image depicts four people, a man in a fur coat, a woman in a dress with a white wig and a black jacket, a man in a brown coat, and a man in a black and white beard.
Question:
What are the effects of the edits? Subject5 is the main character in both the Source Image and the Edited Image.
Subject5:
Someone might mistakenly believe that Subject5 is an actor, he is in a scene of the television show Game of Thrones.

Edits:
The tennis player, racket, crowd of spectators, score board, and man in the background are replaced by a man fishing on the boat in the ocean, with a fish in the boat and a seagull jumping out of the water behind him.
Source Image:
The image depicts a tennis player swinging at a ball with a racket, with a crowd of spectators watching him from the sidelines and a score board in the background. A man is also visible in the background.
Question:
What are the effects of the edits? Subject1 is the main character in both the Source Image and the Edited Image. In the Edited Image, subject1 is the man fishing on the boat in the ocean.
Subject1:
Someone might mistakenly believe that Subject1 is a fisherman rather than an athlete, his pose fits naturally into the setting and the apparatus.

'''
    
    eff_prompt = eff_prompt + "Edits:"
    index = cap["index"]
    edits = extract_edits(edits_lst, index, attr = attr)
    for e in edits:
        eff_prompt = eff_prompt + "\n" + e
    eff_prompt = eff_prompt + "\n" + "Source Image:" + "\n" + cap["global_s"].strip() + " "
    
    eff_prompt = eff_prompt + "\nQuestion:\nWhat are the effects of the edits? "
    
    eff_prompt = eff_prompt + subject.capitalize() + " is the main character in both the Source Image and the Edited Image. "
    
    sub_lst = []
    for s in cap["subject_e"]:
        if s.startswith("subject"):
            sub_lst.append(s.split()[0])
    sub_lst = list(set(sub_lst))

    if subject in sub_lst:
        for s in cap["subject_e"]:
            if s.startswith(subject.lower()):
                eff_prompt = eff_prompt + "In the Edited Image, " + s

    eff_prompt = eff_prompt + "\n" + subject.capitalize() + ":\n"
    # eff_prompt = eff_prompt + "\nEffects:\n"

    # print(eff_prompt)
    
    effects = openai.Completion.create(
                                engine="text-davinci-003",
                                prompt=eff_prompt,
                                max_tokens=512,
                                n=1,
                                stop=None,
                                temperature=0,
                            )

    eff_result = effects.choices[0].text
    
    return eff_result.strip().strip("\n")

# TEST
# import openai
# openai.api_key = "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"
# with open('../Week22_3/Random_cap.json', 'r') as f:
#     data = json.load(f)
# print(get_subject_effect(data[421], attr = "Random Edits", subject = "subject1"))