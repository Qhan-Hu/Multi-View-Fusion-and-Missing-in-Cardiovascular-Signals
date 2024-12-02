import os

### used for self-supervised pretraining, inculuding masked signal modelling and contrastive ECG & PPG ###
inst = r'python main.py'
os.system(inst)

# used for finetuning the pretrained model for blood pressure estimation
inst = r'python main.py with finetune_pulsedb \ per_gpu_batchsize=256 \ dataset_ratio=0.01'
os.system(inst)

# used for finetuning the pretrained model for missing view scenario
inst = r'python main_missing_view.py with finetune_missratio01_ecg \ prompt_length=8 \ exp_name="ablation_prompt_length"'
os.system(inst)

