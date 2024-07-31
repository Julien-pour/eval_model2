import argparse
import copy
import os


# from key import wandb_key   
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM,TrainingArguments
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from tqdm import tqdm
import json
from datasets import Dataset
import portalocker
import random
from time import sleep
from utils import get_formated_chat_dataset
import numpy as np

parser = argparse.ArgumentParser(description="Example script for argument parsing")
# parser.add_argument("--base_path", type=str, help="path to this git project evaluate_model",default="/home/flowers/work/eval_model2/")#)#)
# parser.add_argument("--path_model_base", type=str, help="path where hf model are saved",default="/home/flowers/work/hf/")#)#
 


parser.add_argument("--base_path", type=str, help="path to this git project evaluate_model",default="/gpfswork/rech/imi/uqv82bm/eval_model2/")#"/home/flowers/work/eval_model2/")#)
parser.add_argument("--path_model_base", type=str, help="path where hf model are saved",default="/gpfsscratch/rech/imi/uqv82bm/hf/")#"/home/flowers/work/hf/")#
parser.add_argument("-m", "--arg_model_id", type=str, help=" model",default="deepseek-coder-1.3b-instruct")#"Meta-Llama-3-8B-Instruct")#
parser.add_argument("--path_archive", type=str, help="path where archive for training is",default="archives/aces_elm_seed-5.json")#"archives/rd_gen_seed-5.json")#"archives/preprocess_p3_emb_dedup_puzzles.json")
parser.add_argument("-e", "--arg_epoch", type=int, help="number epoch",default=1)
parser.add_argument("-b", "--arg_bs", type=int, help=" bs train",default=2)
parser.add_argument("-s", "--arg_seed_random", type=int, help="seed for the random generator",default=1)
parser.add_argument("-g", "--arg_gpu", type=str, help="GPU use",default="a100")
parser.add_argument("-a", "--accum_step", type=int, help="number of accumulation step",default=4)
parser.add_argument("--test_base_model", type=str, help="just test base model",default="False")
parser.add_argument("--lr", type=float, help="learning rate",default=1e-6)
parser.add_argument("--name_run", type=str, help="run_name")

n_max_token=2048 #1360*
optim= "paged_adamw_8bit"#"adamw_torch",#"paged_adamw_8bit",#"paged_adamw_32bit",

args = parser.parse_args()
if args.test_base_model == "True":
    print("just test base model")
    exit()
os.environ["WANDB_DISABLED"] = "True"
os.environ['TRANSFORMERS_OFFLINE'] = "1"
os.environ['WANDB_MODE'] = "offline"
os.environ["WANDB_PROJECT"] = "codegpt finetuned"
# os.environ['WANDB_API_KEY'] = wandb_key
os.environ['WANDB_CACHE_DIR'] = args.base_path+"wandb_cache/"
os.environ['TOKENIZERS_PARALLELISM'] = "True"

learning_rate= args.lr
if args.arg_gpu == "v100":
    type_use = torch.float16
    bf16=False
    fp16=True
else:
    type_use = torch.bfloat16
    bf16=True
    fp16=False

# save run config
model_id = args.arg_model_id
params={"name":args.name_run, "lr":args.lr,"epochs":args.arg_epoch,"model_id":model_id} 
try:
    unique_id=f"{os.getenv('SLURM_ARRAY_JOB_ID')}_{os.getenv('SLURM_ARRAY_TASK_ID')}"
except:
    unique_id=f"{os.getenv('SLURM_ARRAY_JOB_ID')}_0"
unique_id=args.path_archive.split("/")[-1].split(".")[0]
filename_save = args.base_path+"save_results/multiple_results/"+f"results_{unique_id}.json"
params["unique_id"]=unique_id


# load dataset
path_train = args.base_path 
path_train += args.path_archive
with open(path_train, encoding="utf-8") as f:
    dataset = json.load(f)
# remove element when "idx_generation" >39
dataset = [d for d in dataset if d["idx_generation"]<=39]

dataset_formated = get_formated_chat_dataset(dataset,text_field="program_str",retun_response=True) # key == chat_data
hf_dir=args.path_model_base
path_load_model=hf_dir+model_id

# init model
tokenizer = AutoTokenizer.from_pretrained(path_load_model,local_files_only=False)
tokenizer.padding_side='right'
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    path_load_model,
    # torch_dtype=type_use,
    # quantization_config=quantization_config,
    device_map="auto",
    local_files_only=True
)
warmup_ratio=0.1
response_template= "Solution 2:"
collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer,mlm=False)
run_name = model_id+unique_id#.split("/")[1]


dat=dataset_formated.map(lambda x: {"formatted_chat": tokenizer.apply_chat_template(x["chat"], tokenize=False, add_generation_prompt=False)},remove_columns=dataset_formated.column_names)
# remove datapoint when they are too long
def filter_long_sequences(example):
    tokens = tokenizer.encode(example['formatted_chat'])
    return len(tokens) <= n_max_token

filtered_dat = dat.filter(filter_long_sequences)

print(f"Original dataset size: {len(dat)}")
print(f"Filtered dataset size: {len(filtered_dat)}")

dat=filtered_dat.shuffle(seed=42) 


lr_scheduler_type= "cosine"
training_arguments=TrainingArguments(
    per_device_train_batch_size=args.arg_bs,
    gradient_accumulation_steps=args.accum_step,
    run_name= run_name,
    save_strategy="no",
    warmup_ratio=warmup_ratio,
    lr_scheduler_type=lr_scheduler_type,
    num_train_epochs=args.arg_epoch,
    learning_rate=learning_rate,
    bf16=bf16, 
    fp16=fp16,
    gradient_checkpointing=False,
    logging_steps=1,
    output_dir="outputs",
    optim=optim, #"adamw_torch",#"adamw_torch",#"paged_adamw_8bit",#"paged_adamw_32bit",
    max_grad_norm=0.3,
    # torch_compile=True
    
)
trainer = SFTTrainer(
    model,#"EleutherAI/gpt-neo-125m",
    tokenizer=tokenizer,
    train_dataset=dat,

    # formatting_func=formatting_prompts_func_,
    dataset_text_field="formatted_chat",
    data_collator=collator,
    max_seq_length=n_max_token,
    args=training_arguments

)


trainer.train()

output_dir = hf_dir+run_name #args.base_path+"hf/datasets"+name # where to save model
trainer.save_model(output_dir)
print("output dir: ",output_dir)


    


