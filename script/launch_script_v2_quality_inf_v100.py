import os
import argparse
import subprocess
from datetime import datetime

seed=1
# script running over epochs, LRs, ratios data
# ncpu 8gpu ->64 
script_1="""#!/bin/bash
#SBATCH --account=imi@v100
#SBATCH -C v100-32g
#SBATCH --job-name={name}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:{n_gpu}
#SBATCH --cpus-per-task=40
{dev_script}

#SBATCH --hint=nomultithread
#SBATCH --time={h}:00:00
#SBATCH --array=0

#SBATCH --output=./out/out_finetune_deep-%A_%a.out
#SBATCH --error=./out/out_finetune_deep-%A_%a.out
module purge

module load python/3.11.5
conda deactivate
module load cuda/12.1.0
conda activate exllama

cd $WORK/eval_model2

epochs=(1)
lrs=(5e-5 1e-6)
ratios=(1.0) # data


"""
script_2="""
MAXWAIT=30
sleep $((RANDOM % MAXWAIT))
conda deactivate
module purge 
module load python/3.11.5

conda activate vllm41
python inference_vllm.py --base_path $WORK/eval_model2/ -e {e} -b 2 --arg_gpu "v100" -a 4 --lr={lr} --test_base_model {test_base_model} --arg_bs_test 1024 --arg_model_id {model_id} -k {k} --n_gpu {n_gpu}
"""

if not os.path.exists('slurm/slurm_files'):
    os.makedirs('slurm/slurm_files')

model_id = "Meta-Llama-3-8B-Instruct"

archive="preprocess_p3_emb_dedup_puzzles.json"
e=1
n_gpu=4
lr=1e-6
test_base_model="True"
dev_script="#SBATCH --qos=qos_gpu-dev"

# data_order= "random"
# for e in [1,2]:
#     for lr in [5e-6, 1e-6]:
#             script_formated = script_1.format(n_gpu=n_gpu,dev_script=dev_script,h="2")+script_2.format(name_archive=archive,e=e,lr=lr,test_base_model=test_base_model,model_id=model_id)
#             extra_path='lr-mode'+str(lr)+'-e-'+str(e)
#             slurmfile_path = f'slurm/run_v100'+extra_path+'.slurm'
#             with open(slurmfile_path, 'w') as f:
#                 f.write(script_formated)
#                 # print(script_formated)
#             subprocess.call(f'sbatch {slurmfile_path}', shell=True)
# dev_script=""
# list_model_id =["deepseek-coder-33b-instruct","deepseek-coder-6.7b-instruct","Phind-CodeLlama-34B-v2","Nous-Hermes-2-Mixtral-8x7B-DPO","c4ai-command-r-v01","c4ai-command-r-plus-GPTQ","Qwen1.5-72B-Chat-GPTQ-Int4","Meta-Llama-3-70B-Instruct","Meta-Llama-3-70B-Instruct-GPTQ","Mixtral-8x22B-Instruct-v0.1-GPTQ-4bit"][-2:-1]
# list_model_name = ["dc33b","dc67b","phind","heres","c4ai","c4aip","qwen","lam70","lam70gptq","mi8x22"][-2:-1]
list_model_id=["Nous-Hermes-2-Mistral-7B-DPO","OpenHermes-2.5-Mistral-7B","Hermes-2-Pro-Mistral-7B"]
list_model_name=["hermesdpo","mistr2.5","mistr-pro"]
list_name_slow_model=["c4aip","qwen","lam70"]
list_name_slowplus_model=["mi8x22"]

for id,model_id in enumerate(list_model_id):
    k=50
    test_base_model="True"
    script_formated = script_1.format(name=list_model_name[id],n_gpu=n_gpu,dev_script=dev_script,h="2")+script_2.format(name_archive=archive,e=e,lr=lr,test_base_model=test_base_model,model_id=model_id,n_gpu=n_gpu,k=k)
    extra_path=model_id+'__base____'
    slurmfile_path = f'slurm/run_v100inf'+extra_path+'.slurm'
    with open(slurmfile_path, 'w') as f:
        f.write(script_formated)
    subprocess.call(f'sbatch {slurmfile_path}', shell=True)