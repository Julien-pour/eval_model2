import os
import argparse
import subprocess
from datetime import datetime

seed=1
# script running over epochs, LRs, ratios data

script_1="""#!/bin/bash
#SBATCH --account=imi@a100
#SBATCH -C a100
#SBATCH --job-name=codellm
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
module purge

module load python/3.11.5
conda deactivate
module load cuda/12.1.0
conda activate exllama

cd $WORK/eval_model2
"""

if not os.path.exists('slurm/slurm_files'):
    os.makedirs('slurm/slurm_files')


    script_2="""
conda deactivate
module purge 
module load python/3.11.5
MAXWAIT=40
sleep $((RANDOM % MAXWAIT))

conda activate vllm532
python inference_vllm.py --base_path $WORK/eval_model2/ --arg_gpu "a100" --test_base_model {test_base_model} --arg_bs_test 8000 --arg_model_id {model_id_2} --seed {seed} --path_archive "archives/{name_archive}" --n_gpu {n_gpu_inference} --path_archive_test "archives/{test_archive}" --file_save_name "difficulty_rebuttal"
"""




list_model_2V100=["Meta-Llama-3-8B-Instruct",
"deepseek-coder-6.7b-instruct",
"deepseek-coder-1.3b-instruct",
"CodeQwen1.5-7B-Chat",
"starcoder2-15b-instruct-v0.1",
"codegemma-7b-it"]

list_model_4V100 = ["Mistral-Large-Instruct-2407-AWQ",
"Meta-Llama-3-70B-Instruct-GPTQ",
"Qwen2-72B-Instruct-AWQ",
"deepseek-coder-33b-instruct",
"Mixtral-8x22B-Instruct-v0.1-GPTQ-4bit"]

list_model_A100 = ["Meta-Llama-3.1-405B-Instruct-AWQ-INT4"]
list_all_model = list_model_A100

list_archive2test = ["aces_elm_seed-5.json","aces_elm_seed-6.json","aces_elm_seed-7.json","Llama-405B_aces_elm_seed-1.json","Mistral-Large-2407_aces_elm_seed-1.json"]
# testset_archive="preprocess_p3_emb_dedup_puzzles.json"

for id_model,model_id in enumerate(list_all_model):
    for path_test_archive in list_archive2test:
        n_gpu=4
        test_base_model="True"
        dev_script="SBATCH --qos=qos_gpu-dev"
        list_archive=[]
        # list_name = ["rd_gen","aces_elm"]
        # list_archive = [i+".json" for i in list_name]
        # list_archive=[]
        # list_seed = [5]
        # for name in list_name:
        #     for seed in list_seed:
                # archive = name+"_seed-"+str(seed)+".json"
                # # name_archive=archive.split(".json")[0]
        script_formated = script_1.format(n_gpu=n_gpu,dev_script=dev_script,h="2")+script_2.format(name_archive=path_test_archive,test_base_model=test_base_model,model_id=model_id,model_id_2=model_id,seed=seed,n_gpu_inference=n_gpu,test_archive=path_test_archive)
        extra_path='lr-mode'+model_id[:7]+"_"+path_test_archive.split(".")[0]
        slurmfile_path = f'slurm/run_a100'+extra_path+'.slurm'
        with open(slurmfile_path, 'w') as f:
            f.write(script_formated)
            # print(script_formated)
        subprocess.call(f'sbatch {slurmfile_path}', shell=True)
