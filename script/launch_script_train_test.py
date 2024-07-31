import os
import argparse
import subprocess
from datetime import datetime

seed=1
# script running over epochs, LRs, ratios data

script_1="""#!/bin/bash
#SBATCH --account=imi@v100
#SBATCH -C v100-32g
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

epochs=(1)
lrs=(5e-5 1e-6)
ratios=(1.0) # data


"""
script_2="""
python finetune.py --base_path $WORK/eval_model2/ --path_archive "archives/{name_archive}" -e {e} -b 4 --arg_gpu "v100" -a 2 --lr={lr} --test_base_model {test_base_model} --arg_model_id {model_id}
conda deactivate
module purge 
module load python/3.11.5
MAXWAIT=40
sleep $((RANDOM % MAXWAIT))

conda activate vllm532
python inference_vllm.py --base_path $WORK/eval_model2/ -e {e} -b 4 --arg_gpu "v100" -a 2 --lr={lr} --test_base_model {test_base_model} --arg_bs_test 1024 --arg_model_id {model_id_2} --seed {seed} --path_archive "archives/{name_archive}" --n_gpu {n_gpu_inference}
"""

if not os.path.exists('slurm/slurm_files'):
    os.makedirs('slurm/slurm_files')

model_id = "deepseek-coder-1.3b-instruct"#"Meta-Llama-3-8B-Instruct"
# testset_archive="preprocess_p3_emb_dedup_puzzles.json"
e=1
n_gpu=4
n_gpu_train=2
test_base_model="False"
dev_script=""#"#SBATCH --qos=qos_gpu-dev"
list_archive=[]

list_name = ["rd_gen","elm","elm_nlp","aces","aces_elm"]
list_archive = [i+".json" for i in list_name]
# data_order= "random"

for e in [1]:
    for lr in [1e-6]:
        seed=-1
        for archive in list_archive:
            # name_archive=archive.split(".json")[0]
            script_formated = script_1.format(n_gpu=n_gpu,dev_script=dev_script,h="2")+script_2.format(name_archive=archive,e=e,lr=lr,test_base_model=test_base_model,model_id=model_id,model_id_2=model_id,seed=seed,n_gpu_inference=n_gpu)
            extra_path='lr-mode'+str(lr)+'-e-'+str(e)
            slurmfile_path = f'slurm/run_v100'+extra_path+'.slurm'
            with open(slurmfile_path, 'w') as f:
                f.write(script_formated)
                # print(script_formated)
            subprocess.call(f'sbatch {slurmfile_path}', shell=True)

list_archive=[]
list_seed = [5, 6, 7]
for name in list_name:
    for seed in list_seed:
        archive = name+"_seed-"+str(seed)+".json"
        # name_archive=archive.split(".json")[0]
        script_formated = script_1.format(n_gpu=n_gpu,dev_script=dev_script,h="2")+script_2.format(name_archive=archive,e=e,lr=lr,test_base_model=test_base_model,model_id=model_id,model_id_2=model_id,seed=seed,n_gpu_inference=n_gpu)
        extra_path='lr-mode'+str(lr)+'-e-'+str(e)
        slurmfile_path = f'slurm/run_v100'+extra_path+'.slurm'
        with open(slurmfile_path, 'w') as f:
            f.write(script_formated)
            # print(script_formated)
        subprocess.call(f'sbatch {slurmfile_path}', shell=True)

name="wizard_gen"
seed=1
archive = name+"_seed-"+str(seed)+".json"
script_formated = script_1.format(n_gpu=n_gpu,dev_script=dev_script,h="2")+script_2.format(name_archive=archive,e=e,lr=lr,test_base_model=test_base_model,model_id=model_id,model_id_2=model_id,seed=seed,n_gpu_inference=n_gpu)
extra_path='lr-mode'+str(lr)+'-e-'+str(e)
slurmfile_path = f'slurm/run_v100'+extra_path+'.slurm'
with open(slurmfile_path, 'w') as f:
    f.write(script_formated)
    # print(script_formated)
subprocess.call(f'sbatch {slurmfile_path}', shell=True)