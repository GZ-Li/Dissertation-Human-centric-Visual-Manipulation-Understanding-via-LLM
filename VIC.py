import sys
sys.path.append("../LAVIS/lavis/")
from models import load_model_and_preprocess
import torch
from PIL import Image
import openai

def extract_q_and_ea(text):
    questions = []
    answers = []
    in_ea = False
    lines = text.split('\n')
    for line in lines:
        line = line.strip()
        if line.startswith("Q:"):
            question = line[2:].strip()
            questions.append(question)
            in_ea = False
        elif line.startswith("EA:"):
            answer = line[3:].strip()
            answers.append(answer)
            in_ea = True
        elif line.startswith("##"):
            in_ea = False
    return questions, answers

def extract_image_text(text):
    parts = text.split("##")
    image_text = [part.strip() for part in parts if part.strip()]
    return image_text

def check_single(index = "88", key_lst = key_lst, edit = "", model = "", vis_processors = "", txt_processors = "", Path = "../Images/dev/", device = torch.device("cpu"), subject = "all the details"):
    log = []
    openai.api_key = key_lst[0]
    source_image_path = Path + str(index) + "_s.jpg"
    edited_image_path = Path + str(index) + "_e.jpg"
    raw_image_s = Image.open(source_image_path).convert("RGB")
    raw_image_e = Image.open(edited_image_path).convert("RGB")
    image_s = vis_processors["eval"](raw_image_s).unsqueeze(0).to(device)
    image_e = vis_processors["eval"](raw_image_e).unsqueeze(0).to(device)
    
    prompt1 = "I have statement about the difference between the source image and the edited image. You can ask question about " + subject + " in the source image and the edited image separately. Try to ask Yes/No question, but you can also ask other questions if necessary. The information should be related to the two images' different part in the statement. Tell me how to validate the difference? And if the statement is true, what are the expected answers to the questions? The statement is written after 'Statement'. Write the questions and expcted answers for the source image after '##Source Image:##', and write the questions and expected answers for the edited image after'##Edited Image:##'. Write each question after 'Q:', and write the expected answers after 'EA:'. The questions and corresponding answers should be proposed one by one. \nStatement: " + edit
    
    QG = openai.Completion.create(
                                    engine="text-davinci-003",
                                    prompt=prompt1,
                                    max_tokens=512,
                                    n=1,
                                    stop=None,
                                    temperature=0,
                                )
    Q = QG.choices[0].text.strip("\n").strip(" ")
    print(Q)
    log.append("To be checked edit: " + edit)
    log.append("Generated Questions for checking/\n" + Q)
    key_lst = key_lst[1:] + [key_lst[0]]
    openai.api_key = key_lst[0]
    image_texts = extract_image_text(Q)
    if ("Source Image" in Q) and ("Edited Image" in Q): 
        source_image_text = image_texts[1]
        edited_image_text = image_texts[3]
    if ("Source Image" in Q) and ("Edited Image" not in Q): 
        source_image_text = image_texts[1]
        edited_image_text = ""
    if ("Source Image" not in Q) and ("Edited Image" in Q): 
        source_image_text = ""
        edited_image_text = image_texts[1]
    source_image_questions, source_image_answers = extract_q_and_ea(source_image_text)
    edited_image_questions, edited_image_answers = extract_q_and_ea(edited_image_text)
    
    #### Source Image QA ####
    s_a_lst = []
    for q in source_image_questions:
        question = txt_processors["eval"](q)
        result_s = model.predict_answers({"image": image_s, "text_input": question}, inference_method = "generate", max_len = 128)[0]
        s_a_lst.append(result_s)
    # s_a_lst = ["Answers to the generated questions for the source image"] + s_a_lst
    log.append(s_a_lst)
    
    #### Edited Image QA ####
    e_a_lst = []
    for q in edited_image_questions:
        question = txt_processors["eval"](q)
        result_e = model.predict_answers({"image": image_e, "text_input": question}, inference_method = "generate", max_len = 128)[0]
        e_a_lst.append(result_e)
    # e_a_lst = ["Answers to the generated questions for the edited image"] + e_a_lst
    log.append(e_a_lst)
        
    #### Check Source Image Answers ####
    s_judge_lst = []
    if len(source_image_answers) != len(source_image_questions):
            truncate_num = min(len(source_image_answers), len(source_image_questions))
            source_image_answers = source_image_answers[: truncate_num]
            source_image_questions = source_image_questions[: truncate_num]
    for q in range(len(source_image_questions)):
        prompt2 = "I have a question, a reference answer and a candidate answer. Help me to judge whether the candidate answer matches the reference answer or not. If the reference answer is correct, return [[YES]]. If the reference answer is not correct, return [[NO]]. For Yes/No question, just focus on yes or no. The question is written after 'Question:', the reference answer is written after 'Reference Answer:', and the candidate answer is written after 'Candidate Answer:'. Write your answer after 'Answer:'.\nQuestion: " + source_image_questions[q] + "\nReference Answer: " + source_image_answers[q] + "\nCandidate Answer: " + s_a_lst[q] + "\nAnswer: "
        print(prompt2)
        Judge = openai.Completion.create(
                                            engine="text-davinci-003",
                                            prompt=prompt2,
                                            max_tokens=512,
                                            n=1,
                                            stop=None,
                                            temperature=0,
                                        )
        J = Judge.choices[0].text.strip("\n").strip(" ")
        print(J)
        log.append("Answer Consistent? For source image: " + J)
        key_lst = key_lst[1: ] + [key_lst[0]]
        openai.api_key = key_lst[0]
        if "[[YES]]" in J:
            s_judge_lst.append("True")
        else:
            if "[[NO]]" in J:
                s_judge_lst.append("False")
            else:
                s_judge_lst.append("Error")
        print(s_judge_lst)
    # s_judge_lst = ["Answer from source image consistent?"] + s_judge_lst
    log.append(s_judge_lst)
    
    #### Check Edited Image Answers ####    
    e_judge_lst = []
    if len(edited_image_answers) != len(edited_image_questions):
            truncate_num = min(len(edited_image_answers), len(edited_image_questions))
            edited_image_answers = edited_image_answers[: truncate_num]
            edited_image_questions = edited_image_questions[: truncate_num]
    for q in range(len(edited_image_questions)):
        prompt2 = "I have a question, a reference answer and a candidate answer. Help me to judge whether the candidate answer matches the reference answer or not. If the reference answer is correct, return [[YES]]. If the reference answer is not correct, return [[NO]]. For Yes/No question, just focus on yes or no. Focus on the important part of the answers, and you can ignore some tiny differences on details. The question is written after 'Question:', the reference answer is written after 'Reference Answer:', and the candidate answer is written after 'Candidate Answer:'. Write your answer after 'Answer:'.\nQuestion: " + edited_image_questions[q] + "\nReference Answer: " + edited_image_answers[q] + "\nCandidate Answer: " + e_a_lst[q] + "\nAnswer: "
        Judge = openai.Completion.create(
                                            engine="text-davinci-003",
                                            prompt=prompt2,
                                            max_tokens=512,
                                            n=1,
                                            stop=None,
                                            temperature=0,
                                        )
        J = Judge.choices[0].text.strip("\n").strip(" ")
        print(J)
        log.append("Answer Consistent? For edited image: " + J)
        key_lst = key_lst[1: ] + [key_lst[0]]
        openai.api_key = key_lst[0]
        if "[[YES]]" in J:
            e_judge_lst.append("True")
        else:
            if "[[NO]]" in J:
                e_judge_lst.append("False")
            else:
                e_judge_lst.append("Error")
        print(e_judge_lst)
    # e_judge_lst = ["Answer from edited image consistent?"] + e_judge_lst
    log.append(e_judge_lst)
                
    if ((s_judge_lst.count("False") == len(s_judge_lst)) and (len(s_judge_lst) != 0)) or ((e_judge_lst.count("False") == len(e_judge_lst) and (len(e_judge_lst) != 0))):
        log.append("Reject!!!!!")
        return False, "", key_lst, log
    
    else:
        if ((s_judge_lst.count("True") + s_judge_lst.count("Error")) == len(s_judge_lst)) and ((e_judge_lst.count("True") + e_judge_lst.count("Error")) == len(e_judge_lst)):
            log.append("Accept!!!!!")
            return True, edit, key_lst, log
        else:
            log.append("Adjust!!!!!")
            prompt3 = "Please remove the information related to the given question from the given statement. But make sure there are no grammar mistakes in the sentence after removing. If you can't return a fluent sentence, just return "". The question is written after 'Question: ', and the statement is written after 'Statement: '. \nQuestion: "
            for q in range(len(source_image_questions)):
                if s_judge_lst[q] == "False":
                    prompt3 = prompt3 + source_image_questions[q] + " "
            for q in range(len(edited_image_questions)):
                if e_judge_lst[q] == "False":
                    prompt3 = prompt3 + edited_image_questions[q] + " "
            prompt3 = prompt3 + "\nStatement: " + edit
            # print(prompt3)
            Remove = openai.Completion.create(
                                                engine="text-davinci-003",
                                                prompt=prompt3,
                                                max_tokens=512,
                                                n=1,
                                                stop=None,
                                                temperature=0,
                                            )
            R = Remove.choices[0].text.strip().strip("\n").strip(" ")
            print(R)
            key_lst = key_lst[1: ] + [key_lst[0]]
            openai.api_key = key_lst[0]
            return True, R, key_lst, log
 
def check(index = "88", key_lst = [], edits = [], model = "", vis_processors = "", txt_processors = "", Path = "../Images/dev/", device = torch.device("cpu"), subject = "all the details"):
    logs = []
    checked_edits = []
    for edit in edits:
        print(edit)
        valid, checked_edit, key_lst, log = check_single(index = index, key_lst = key_lst, edit = edit, model = model, vis_processors = vis_processors, txt_processors = txt_processors, Path = Path, device = device, subject = subject)
        logs.append(log)
        if valid == True:
            checked_edits.append(checked_edit)
    if len(checked_edits) == 0:
        valid = False
    else:
        valid = True
    return valid, checked_edits, key_lst, logs
 
#### TEST ####  
# if __name__ == '__main__':     
#     Path = "../Images/dev/"
#     index = "88"
#     device = torch.device("cpu")
#     source_image_path = Path + str(index) + "_s.jpg"
#     edited_image_path = Path + str(index) + "_e.jpg"
#     raw_image_s = Image.open(source_image_path).convert("RGB")
#     raw_image_e = Image.open(edited_image_path).convert("RGB")
#     model, vis_processors, txt_processors = load_model_and_preprocess(name="blip_vqa", model_type="vqav2", is_eval=True, device=device)

#     edits = ["The man in the source image is sitting in a chair, while in the edited image he is sitting at a table.", "In the source image, there is money falling from the sky in front of him, while in the edited image money is coming out of his shoe.", "In the source image, there is a blue wall with a FIFA logo behind him, while in the edited image there is a woman's leg with a cigarette in a glass of water in the foreground.", "In the source image, there is a bottle of money in the air and a man in front of the money in the air, while in the edited image there is a cigarette in the air and in front of his hand."]
#     edit = edits[2]
#     edits = ["In the source image, there is a blue wall with a FIFA logo behind him, while in the edited image there is a woman's leg with a cigarette in a glass of water in the foreground."]

#     print(check(index = "88", key_lst = key_lst, edits = edits, model = model, vis_processors = vis_processors, txt_processors = txt_processors, Path = "../Images/dev/"))

#     print(check_single(edit = edits[0], index = "88", key_lst = key_lst, model = model, vis_processors = vis_processors, txt_processors = txt_processors, Path = "../Images/dev/"))