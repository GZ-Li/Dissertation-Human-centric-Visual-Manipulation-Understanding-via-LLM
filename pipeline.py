from VIC import check
import sys
sys.path.append("../LAVIS/lavis/")
from models import load_model_and_preprocess
import torch
from PIL import Image
import openai
import re

#### Notice ####
#### The input subject:
#### if it's a word or a phrase, remember to add article to ensure the prompt's fluency;
#### if it's a W-question, just directly input;
def get_edits(index = "88", Path = "../Images/dev/", device = torch.device("cuda:1"), key_lst = [], k = 3, subject = "all the detils"):
    
    openai.api_key = key_lst[0]
    q_lst = []
    a_s_lst = []
    a_e_lst = []
    log = {}
    pattern = r'\d+\.\s'
    model, vis_processors, txt_processors = load_model_and_preprocess(name="blip_caption", model_type="large_coco", is_eval=True, device=device)
    qa_model, qa_vis_processors, qa_txt_processors = load_model_and_preprocess(name="blip_vqa", model_type="vqav2", is_eval=True, device=device)
    img1 = Image.open(Path + index + "_s.jpg").convert("RGB").save("source.jpg")
    img2 = Image.open(Path + index + "_e.jpg").convert("RGB").save("edit.jpg")
    img_s = Image.open("source.jpg").convert('RGB')
    img_e = Image.open("edit.jpg").convert('RGB')

    global_s = model.generate({"image": vis_processors["eval"](img_s).unsqueeze(0).to(device)}, use_nucleus_sampling=False, num_captions=1, min_length = 64, max_length = 256, repetition_penalty = 1.0, top_p = 0.1)[0]
    global_e = model.generate({"image": vis_processors["eval"](img_e).unsqueeze(0).to(device)}, use_nucleus_sampling=False, num_captions=1, min_length = 64, max_length = 256, repetition_penalty = 1.0, top_p = 0.1)[0]
    # print(global_s)
    # print(global_e)
    log["original_source_caption"] = global_s
    log["original_edited_caption"] = global_e
    F_S = global_s
    F_E = global_e
    
    F_prompt = "Extract the information related to " + subject + " in the picture from the given caption. The caption is written after 'Caption:'. Give your answer after 'Answer: '.\nCaption: " + global_s + "\nAnswer: "
    FI = openai.Completion.create(
                                    engine="text-davinci-003",
                                    prompt=F_prompt,
                                    max_tokens=512,
                                    n=1,
                                    stop=None,
                                    temperature=0,
                                )
    F_S = FI.choices[0].text
    # print("F_S:{}".format(F_S))
    key_lst = key_lst[1:] + [key_lst[0]]
    openai.api_key = key_lst[0]
    F_prompt = "Extract the information related to " + subject + " in the picture from the given caption. The caption is written after 'Caption:'. Give your answer after 'Answer: '.\nCaption: " + global_e + "\nAnswer: "
    FI = openai.Completion.create(
                                    engine="text-davinci-003",
                                    prompt=F_prompt,
                                    max_tokens=512,
                                    n=1,
                                    stop=None,
                                    temperature=0,
                                )
    F_E = FI.choices[0].text
    # print("F_E:{}".format(F_E))
    key_lst = key_lst[1:] + [key_lst[0]]
    openai.api_key = key_lst[0]
    log["encoded_source_visual_information"] = F_S
    log["encoded_edited_visual_information"] = F_E
    
    RS_prompt = "Tell me the difference about " + subject + " in the source image and the edited image, according to the given captions of the source image and the edited image. If you can detect some differences, express them as how the source image is changed to the edited image. If you can't detect any difference, just return [[NO]], and don't imagine. The caption of the source image is written after 'Source:' and the caption of the edited image is written after 'Edited'. Label the edits with numbers like '1.', '2.'. \nSource: " + F_S.strip("\n").strip(" ") + "\nEdited:" + F_E.strip("\n").strip(" ")
    # print(RS_prompt)
    RS = openai.Completion.create(
                                    engine="text-davinci-003",
                                    prompt=RS_prompt,
                                    max_tokens=512,
                                    n=1,
                                    stop=None,
                                    temperature=0,
                                )
    E = RS.choices[0].text.strip("\n").strip(" ")
    # print(E)
    key_lst = key_lst[1:] + [key_lst[0]]
    openai.api_key = key_lst[0]
    E_lst = re.split(pattern, E)[1: ]
    E_lst = [edit.strip("\n").strip(" ") for edit in E_lst]
    log["checking_log"] = []
    log["unchecking edits"] = E_lst
    # print("********** START CHECKING **********")
    _, E_lst, key_lst, c_log = check(index = index, key_lst = key_lst, edits = E_lst, model = qa_model, vis_processors = qa_vis_processors, txt_processors = qa_txt_processors, device = device, subject = subject)
    log["checking_log"].append(c_log)
    
    if len(E_lst) == 0:
        counter = 0
        while True:
            sup_prompt = "I have captions of two images, but the information related to " + subject + " is not sufficient enough. Help me to generate a question, whose answer supplement more details and information about " + subject + ". Don't return Yes or No questions. Don't return questions that have already asked. The asked questions are written after 'Log:'. The caption of the source image is written after 'Source Image:', and the caption of the edited image is written after 'Edited Image: '.\nSource Image: " + F_S + "\nEdited Image: " + F_E
            if len(q_lst) > 0:
                sup_prompt = sup_prompt + "\nLog: "
                for q in q_lst:
                    sup_prompt = sup_prompt + q + " "
            else:
                sup_prompt = sup_prompt + "\nLog: "
            S = openai.Completion.create(
                                            engine="text-davinci-003",
                                            prompt=sup_prompt,
                                            max_tokens=512,
                                            n=1,
                                            stop=None,
                                            temperature=0,
                                        )
            GQ = S.choices[0].text.strip("\n").strip(" ").strip("\n")
            # print(GQ)
            key_lst = key_lst[1:] + [key_lst[0]]
            openai.api_key = key_lst[0]
            q_lst.append(GQ)
            
            qa_s_image = qa_vis_processors["eval"](img_s).unsqueeze(0).to(device)
            qa_e_image = qa_vis_processors["eval"](img_e).unsqueeze(0).to(device)
            question = qa_txt_processors["eval"](GQ)
            answer_s = qa_model.predict_answers({"image": qa_s_image, "text_input": question}, inference_method = "generate")[0]
            answer_e = qa_model.predict_answers({"image": qa_e_image, "text_input": question}, inference_method = "generate")[0]
            # print(answer_s)
            # print(answer_e)
            a_s_lst.append(answer_s)
            a_e_lst.append(answer_e)
            
            v_prompt = "Help me to judge if the statement answers the question? If the statement answers the question, just return [[YES]]. If the statement doesn't answer the question, just return [[NO]]. The question is written after 'Question'. The statement is written after 'Statement: '."
            s_v_prompt = v_prompt + "\nQuestion: " + GQ + "\nStatement: " + answer_s
            e_v_prompt = v_prompt + "\nQuestion: " + GQ + "\nStatement: " + answer_e
            S_V = openai.Completion.create(
                                            engine="text-davinci-003",
                                            prompt=s_v_prompt,
                                            max_tokens=512,
                                            n=1,
                                            stop=None,
                                            temperature=0,
                                        )
            s_v = S_V.choices[0].text.strip("\n").strip(" ")
            key_lst = key_lst[1:] + [key_lst[0]]
            openai.api_key = key_lst[0]
            # print(s_v)
            E_V = openai.Completion.create(
                                            engine="text-davinci-003",
                                            prompt=e_v_prompt,
                                            max_tokens=512,
                                            n=1,
                                            stop=None,
                                            temperature=0,
                                        )
            e_v = E_V.choices[0].text.strip("\n").strip(" ")
            key_lst = key_lst[1:] + [key_lst[0]]
            openai.api_key = key_lst[0]
            # print(e_v)
            
            sum_prompt = "Combine the information within the question, answer and the information within the given caption. Don't add information. Don't miss any information. The caption is written after 'Caption:'. The question is written after 'Question:', and the corresponding answer is written after 'Answer:'."
            s_sum_prompt = sum_prompt + "\nCaption: " + F_S + "\nQuestion: " + GQ + "\nAnswer: " + answer_s
            e_sum_prompt = sum_prompt + "\nCaption: " + F_E + "\nQuestion: " + GQ + "\nAnswer: " + answer_e
            if "[[YES]]" in s_v:
                S_SUM = openai.Completion.create(
                                                engine="text-davinci-003",
                                                prompt=s_sum_prompt,
                                                max_tokens=512,
                                                n=1,
                                                stop=None,
                                                temperature=0,
                                            )
                F_S = S_SUM.choices[0].text.strip("\n").strip(" ")
                # print(F_S)
                log["encoded_source_visual_information"] = F_S
                key_lst = key_lst[1:] + [key_lst[0]]
                openai.api_key = key_lst[0]
            if "[[YES]]" in e_v:
                E_SUM = openai.Completion.create(
                                                engine="text-davinci-003",
                                                prompt=e_sum_prompt,
                                                max_tokens=512,
                                                n=1,
                                                stop=None,
                                                temperature=0,
                                            )
                F_E = E_SUM.choices[0].text.strip("\n").strip(" ")
                # print(F_E)
                log["encoded_edited_visual_information"] = F_E
                key_lst = key_lst[1:] + [key_lst[0]]
                openai.api_key = key_lst[0]
            RRS_prompt = "Tell me the difference about " + subject + " in the source image and the edited image, according to the given captions of the source image and the edited image. If you can detect some differences, express them as how the source image is changed to the edited image. If you can't detect any difference, just return [[NO]], and don't imagine. Give your answer after 'Answer:' The caption of the source image is written after 'Source:' and the caption of the edited image is written after 'Edited'. Label the edits with numbers like '1.', '2.'. \nSource: " + F_S.strip("\n").strip(" ") + "\nEdited:" + F_E.strip("\n").strip(" ") + "\nAnswer: "
            RS = openai.Completion.create(
                                            engine="text-davinci-003",
                                            prompt=RS_prompt,
                                            max_tokens=512,
                                            n=1,
                                            stop=None,
                                            temperature=0,
                                        )
            E = RS.choices[0].text.strip("\n").strip(" ")
            key_lst = key_lst[1:] + [key_lst[0]]
            openai.api_key = key_lst[0]
            E_lst = re.split(pattern, E)[1: ]
            E_lst = [edit.strip("\n").strip(" ") for edit in E_lst]
            log["unchecking edits"] = E_lst
            _, E_lst, key_lst, c_log = check(index = index, key_lst = key_lst, edits = E_lst, model = qa_model, vis_processors = qa_vis_processors, txt_processors = qa_txt_processors, device = device, subject = subject)
            log["checking_log"].append(c_log)
            # print(E)
            key_lst = key_lst[1:] + [key_lst[0]]
            openai.api_key = key_lst[0]
            if len(E_lst) > 0:
                break
            else:
                counter = counter + 1
                if counter == k + 1:
                    log["Edits"] = []
                    log["question_lst"] = q_lst
                    log["answer_source_lst"] = a_s_lst
                    log["answer_edited_lst"] = a_e_lst
                    return "No Difference!", q_lst, a_s_lst, a_e_lst, log, key_lst
    log["Edits"] = E_lst
    log["question_lst"] = q_lst
    log["answer_source_lst"] = a_s_lst
    log["answer_edited_lst"] = a_e_lst
    return E_lst, q_lst, a_s_lst, a_e_lst, log, key_lst
    
    

####### TEST #######
# if __name__ == '__main__':
#     with open("../Week16/key_3.txt", "r") as f:
#         content = f.read()
#     key_lst = content.split("\n")
#     print(get_edits(index = "891", key_lst=key_lst)) 
#     index = "88"; subject = "the man in the suit", "the background", "how does the man feel"
#     index = "891"; subject = "the whole picture", "Trump", "background", "the mental state of the woman", "what is the man doing"