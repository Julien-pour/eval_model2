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
export TMPDIR=$JOBSCRATCH

cd $WORK/eval_model2
"""
script_2="""
MAXWAIT=40
sleep $((RANDOM % MAXWAIT))
module purge

module load python/3.11.5

conda deactivate

conda activate vllm41
python inference_vllm_test_archives.py --base_path $WORK/eval_model2/ --arg_gpu "v100" --arg_bs_test 10000 --arg_model_id {model_id} -k {k} --n_gpu {n_gpu} --eager_mode {eager_mode} --path_archive {path_archive} 
"""

if not os.path.exists('slurm/slurm_files'):
    os.makedirs('slurm/slurm_files')

model_id = "Meta-Llama-3-8B-Instruct"
dev_scipt_mode=False

n_gpu=4
dev_script="#SBATCH --qos=qos_gpu-dev"

list_name=["rd_gen","elm","elm_nlp","aces","aces_smart","aces_smart_diversity","aces_smart_elm","aces_smart_elm_diversity","aces_diversity","aces_elm","aces_elm_diversity"]
list_seed=[5,6,7]
path_base="/gpfsscratch/rech/imi/uqv82bm/archives/last_test/"
template="{path_base}{name}_seed-{seed}.json"

list_model_id=["deepseek-coder-1.3b-instruct","deepseek-coder-6.7b-instruct","CodeQwen1.5-7B-Chat","Mixtral-8x7B-Instruct-v0.1","deepseek-coder-33b-instruct","Mixtral-8x22B-Instruct-v0.1-GPTQ-4bit","Phind-CodeLlama-34B-v2","Mistral-7B-Instruct-v0.1","Mistral-7B-Instruct-v0.2","WizardCoder-33B-V1.1","Meta-Llama-3-8B-Instruct","Meta-Llama-3-70B-Instruct-GPTQ","Phi-3-mini-4k-instruct"]
list_model_name=["dp1b","dp6b","codeqwen15","mixtral8x7b","dc33b","mi8x22","phind","mistral7b1","mistral7b2","WizardCoder33B","Llama3-8B","Llama3-70B","phi3"]
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
        h="20"
    else:
        eager_mode="False"
        h="5"
    k=10
    for name in list_name:
        for seed in list_seed:
            path_archive=template.format(path_base=path_base,name=name,seed=seed)
            script_formated = script_1.format(name=list_model_name[id],n_gpu=n_gpu,dev_script=dev_script,h=h)+script_2.format(path_archive=path_archive,model_id=model_id,n_gpu=n_gpu,k=k,eager_mode=eager_mode)
            extra_path=model_id+'__base____'+name
            slurmfile_path = f'slurm/run_v100inf'+extra_path+'.slurm'
            with open(slurmfile_path, 'w') as f:
                f.write(script_formated)
            subprocess.call(f'sbatch {slurmfile_path}', shell=True)