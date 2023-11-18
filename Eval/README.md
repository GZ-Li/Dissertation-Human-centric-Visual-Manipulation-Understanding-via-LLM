# Automatic Evaluation

EMU (Edited Media Understanding) dataset doesn't provide the ground truths of manipulations. Hence, we map the manipulations to four downstream tasks (implications, intents, mental states, and effects) whose ground truths are accessible.

![image](https://github.com/GZ-Li/Dissertation-Human-centric-Visual-Manipulation-Understanding-via-LLM/blob/main/Eval/Automatic%20Evaluation.png)

'implication.py', 'intent.py', 'mental.py', and 'effect.py' implement how we can get the four downstream tasks based on the detected manipulation returned by '../pipeline_main.py'. And the detected manipulations are expected to be stored in the 'Guided_Edits.json'. 'get_pipeline_response.py' can be called for getting all cases' predictions on the four downstream tasks:

```bash
python prompt_captioning.py
```
