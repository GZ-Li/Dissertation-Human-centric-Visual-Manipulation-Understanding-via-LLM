# Dissertation-Human-centric-Visual-Manipulation-Understanding-via-LLM

This repository contains data and code for the submitted dissertation: _Human-centric Visual Manipulation Understanding via LLM_.

![image](https://github.com/GZ-Li/Dissertation-Human-centric-Visual-Manipulation-Understanding-via-LLM/blob/main/pipeline.png)

We propose a pipeline containing two core methods: Visual Information Checking (VIC) and Taxonomy Guideline. VIC is implemented as the 'VIC.py', and the details of the Taxonomy Guideline can be found in 'pipeline.py'. 

To generate the description of the manipulation between a given image pair, you can call the *get_edits* function in 'pipeline.py' as:

```bash
from pipeline import get_edits
```
The necessary input of this function includes the path of the image pair (or index), the focal point (defaulting to 'all the details'), the maximum iteration times when the pipeline cannot detect any correct manipulation, and a list of keys to call OpenAI API.

'pipeline_main.py' can generate the detected manipulations of all the cases in the given dataset. It can executed by:

```bash
python pipeline_main.py
```
But the dataset should be applied [here](https://jeffda.com/edited-media-understanding) and stored in an exclusive folder called "EMU_Dataset". Besides, the corresponding images need crawling from the URLs provided in the EMU dataset and stored in an exclusive folder called "Images". The environment where it runs should be configured with [BLIP](https://github.com/salesforce/LAVIS).
