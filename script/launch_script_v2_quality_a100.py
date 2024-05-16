import os
import argparse
import subprocess
from datetime import datetime

seed=1
# script running over epochs, LRs, ratios data
script_1="""#!/bin/bash
#SBATCH --account=imi@a100
#SBATCH -C a100
#SBATCH --job-name={name}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:{n_gpu}
#SBATCH --cpus-per-task=16
{dev_script}
#SBATCH --hint=nomultithread
#SBATCH --time={h}:00:00
#SBATCH --array=0

#SBATCH --output=./out/out_finetune_deep-%A_%a.out
#SBATCH --error=./out/out_finetune_deep-%A_%a.out

module purge
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
python finetune.py --base_path $WORK/eval_model2/ --path_archive "archives/{name_archive}" -e {e} -b 2 --arg_gpu "a100" -a 4 --lr={lr} --test_base_model {test_base_model}
conda deactivate
module purge 
module load python/3.11.5

conda activate vllm41
python inference_vllm.py --base_path $WORK/eval_model2/ -e {e} -b 2 --arg_gpu "a100" -a 4 --lr={lr}  --test_base_model {test_base_model} --arg_bs_test 1024 --arg_model_id {model_id}
"""

if not os.path.exists('slurm/slurm_files'):
    os.makedirs('slurm/slurm_files')

model_id = "Meta-Llama-3-8B-Instruct"

archive="preprocess_p3_emb_dedup_puzzles.json"
e=1
n_gpu=4
test_base_model="False"
# data_order= "random"
dev=False
dev_script="#SBATCH --qos=qos_gpu-dev"

for e in [1,2]:
    for lr in [5e-6, 1e-6]:
            script_formated = script_1.format(n_gpu=n_gpu,dev_script=dev_script,h="2")+script_2.format(name_archive=archive,e=e,lr=lr,test_base_model=test_base_model,model_id=model_id)
            extra_path='lr-mode'+str(lr)+'-e-'+str(e)
            slurmfile_path = f'slurm/run_a100'+extra_path+'.slurm'
            with open(slurmfile_path, 'w') as f:
                f.write(script_formated)
                # print(script_formated)
            subprocess.call(f'sbatch {slurmfile_path}', shell=True)
dev_script=""
list_model_id = ["deepseek-coder-33b-instruct","deepseek-coder-6.7b-instruct","Phind-CodeLlama-34B-v2","Nous-Hermes-2-Mixtral-8x7B-DPO","c4ai-command-r-v01","c4ai-command-r-plus-GPTQ","Qwen1.5-72B-Chat-GPTQ-Int4","Meta-Llama-3-70B-Instruct","Meta-Llama-3-70B-Instruct-GPTQ","Mixtral-8x22B-Instruct-v0.1-GPTQ-4bit"]
list_model_name =["dc33b","dc67b","phind","hermes","c4ai","c4aip","qwen","lam70","lam70gptq","mi8x22"]
for id,model_id in enumerate(list_model_id):
    test_base_model="True"
    script_formated = script_1.format(name=list_model_name[id],n_gpu=n_gpu,dev_script=dev_script,h="10")+script_2.format(name_archive=archive,e=e,lr=lr,test_base_model=test_base_model,model_id=model_id)
    extra_path=model_id+'__base____'
    slurmfile_path = f'slurm/run_a100inf'+extra_path+'.slurm'
    with open(slurmfile_path, 'w') as f:
        f.write(script_formated)
    subprocess.call(f'sbatch {slurmfile_path}', shell=True)

