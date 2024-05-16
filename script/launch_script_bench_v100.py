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
python inference_vllm_test_archives.py --base_path $WORK/eval_model2/ --name_archive {name_archive} --arg_gpu "v100" --arg_bs_test 2048 --arg_model_id {model_id} -k {k} --n_gpu {n_gpu} --eager_mode {eager_mode}
"""

if not os.path.exists('slurm/slurm_files'):
    os.makedirs('slurm/slurm_files')

model_id = "Meta-Llama-3-8B-Instruct"
dev_scipt_mode=True
archive="preprocess_p3_emb_dedup_puzzles.json"
n_gpu=4
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
list_model_id=["deepseek-coder-1.3b-instruct","deepseek-coder-6.7b-instruct","CodeQwen1.5-7B-Chat","Mixtral-8x7B-Instruct-v0.1","deepseek-coder-33b-instruct","Mixtral-8x22B-Instruct-v0.1-GPTQ-4bit","Phind-CodeLlama-34B-v2","Mistral-7B-Instruct-v0.1","Mistral-7B-Instruct-v0.2","WizardCoder-33B-V1.1","Meta-Llama-3-8B-Instruct","Meta-Llama-3-70B-Instruct"]
list_model_name=["dp1b","dp6b","codeqwen15","mixtral8x7b","dc33b","mi8x22","phind","mistral7b1","mistral7b2","WizardCoder33B","Llama3-8B","Llama3-70B"]
list_name_slow_model=["c4aip","qwen","lam70"]
list_name_eager_model=["mi8x22","mixtral8x7b"]
if dev_scipt_mode:
    dev_script="#SBATCH --qos=qos_gpu-dev"
    h="2"
else:
    dev_script=""
    h="10"
for id,model_id in enumerate(list_model_id):
    if list_model_name[id] in list_name_eager_model:
        eager_mode="True"
    else:
        eager_mode="False"
    k=20
    test_base_model="True"
    script_formated = script_1.format(name=list_model_name[id],n_gpu=n_gpu,dev_script=dev_script,h=h)+script_2.format(name_archive=archive,model_id=model_id,n_gpu=n_gpu,k=k,eager_mode=eager_mode)
    extra_path=model_id+'__base____'
    slurmfile_path = f'slurm/run_v100inf'+extra_path+'.slurm'
    with open(slurmfile_path, 'w') as f:
        f.write(script_formated)
    subprocess.call(f'sbatch {slurmfile_path}', shell=True)