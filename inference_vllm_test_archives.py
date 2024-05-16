from vllm import LLM,SamplingParams
import argparse
import copy
import os
from tqdm import tqdm
import json
import portalocker
import random
from time import sleep
import numpy as np
from code_sandbox import evaluate
os.environ['TOKENIZERS_PARALLELISM'] = "True"

from utils import pass_at_k,judge_parallel,get_formated_chat_dataset
parser = argparse.ArgumentParser(description="Example script for argument parsing")
parser.add_argument("--base_path", type=str, help="path to this git project evaluate_model",default="/gpfswork/rech/imi/uqv82bm/eval_model2/")
parser.add_argument("--path_model_base", type=str, help="path where hf model are saved",default="/gpfsscratch/rech/imi/uqv82bm/hf/")
parser.add_argument("-m", "--arg_model_id", type=str, help=" model",default="Meta-Llama-3-8B-Instruct")
parser.add_argument( "--arg_bs_test", type=int, help=" bs train",default=1024)
parser.add_argument("--name_archive", type=str, help="name archive to test")

parser.add_argument("-s", "--arg_seed_random", type=int, help="seed for the random generator",default=1)
parser.add_argument("-g", "--arg_gpu", type=str, help="GPU use",default="v100")
parser.add_argument("--name_run", type=str, help="run_name")
parser.add_argument("-k", "--arg_k", type=int, help="k in pass@k",default=50)
parser.add_argument("--n_gpu", type=int, help="how many gpu to use",default=2)
parser.add_argument("--eager_mode", type=str, help="eager_mode",default="False")
parser.add_argument("--swap_space", type=float, help="swap space",default=1)

n_max_token=2048 #1360*

args = parser.parse_args()
eager_mode=args.eager_mode.lower()=="true"
learning_rate= args.lr
model_id =   args.arg_model_id


accum_step=args.accum_step

# name: name of the methode (aces,elm-nlp,aces)
params={"model_id":model_id,"name_archive":args.name_archive}


try:
    unique_id=f"{os.getenv('SLURM_ARRAY_JOB_ID')}_{os.getenv('SLURM_ARRAY_TASK_ID')}"
except:
    unique_id=f"{os.getenv('SLURM_ARRAY_JOB_ID')}_0"
filename_save = args.base_path+"save_results/multiple_results/"+f"good_results_{unique_id}.json"
params["unique_id"]=unique_id

run_name = model_id+unique_id

    
name_json_save_all = args.base_path+f"save_results/benchmark_archives.json"#.split("/")[1]

name_json_save_all_solution = args.base_path+f"save_results/save_sol/{model_id}_benchmark.json"#.split("/")[1]


name_json_save_speed = args.base_path+f"save_results/speed_search_llama.json"

if not os.path.exists(name_json_save_speed):
    # Create a new JSON file with some sample data
    sample_data = {}
    with open(name_json_save_speed, 'w') as file:
        json.dump(sample_data, file, indent=4)

if not os.path.exists(name_json_save_all):
    # Create a new JSON file with some sample data
    sample_data = {}
    with open(name_json_save_all, 'w') as file:
        json.dump(sample_data, file, indent=4)
if not os.path.exists(name_json_save_all_solution):
    # Create a new JSON file with some sample data
    sample_data = {}
    with open(name_json_save_all_solution, 'w') as file:
        json.dump(sample_data, file, indent=4)


hf_dir=args.path_model_base


output_dir = hf_dir+run_name


if args.test_base_model.lower()=="true":
    output_dir = hf_dir+model_id
if args.arg_gpu=="v100" or "gptq" in model_id.lower():
    dtype="half"
else:
    dtype="auto"
# if "Mixtral" in model_id:
#     enforce_eager=True
# else:
#     enforce_eager=False
from transformers import AutoTokenizer
tokenizer=AutoTokenizer.from_pretrained(output_dir)


llm = LLM(output_dir,max_model_len=2048,enforce_eager=eager_mode,tensor_parallel_size=args.n_gpu,dtype=dtype,swap_space=args.swap_space)
sampling_params = SamplingParams(n=args.arg_k,
            temperature=0.8,
            top_p=1,
            max_tokens=1024,
            # presence_penalty=1.15,
            # stop_token_ids=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
            
        )


# testset= preprocessing_P3_no_test(split="test",n_token_max=1024,path=args.base_path,tokenizer=tokenizer)
path_test=args.base_path+"archives/"+args.name_archive#"P3_test_emb_wizard3B.json"
with open(path_test,mode = "r") as f:
    testset = json.load(f)

testset = [x for x in testset if (x["idx_generation"]<=130 and x["idx_generation"]>=0)]
testset_f = [p["program_str"].split("def g")[0].strip() for p in testset]
curr_idx=0
correct_puzz=0

num_return_sequences=args.arg_k #n_try
list_all_passk=[[] for i in range(num_return_sequences)]
list_passk=[]

list_puzzle=[]

    
# list_testset= [x["program_str"] for x in testset]
dat_chat = get_formated_chat_dataset(testset,text_field="program_str",retun_response=False,return_hf_data=False)
list_testset= []


for chat in dat_chat:
    for i in range(len(chat["chat"])):
        idx_to_del=[]
        if "Mistral-7B-Instruct-v0.1" in model_id or "Mistral-7B-Instruct-v0.2" in model_id:
            if chat["chat"][i]["role"]=="system":
                idx_to_del.append(i)
                # del chat["chat"][i]
        if chat["chat"][i]["role"]=="assistant":
            idx_to_del.append(i)
            # del chat["chat"][i]
    for idx_del in idx_to_del[::-1]:
        del chat["chat"][idx_del]
    chat_instruction = tokenizer.apply_chat_template(chat["chat"], tokenize=False, add_generation_prompt=True)
    list_testset.append(chat_instruction)


str_to_add=str(
    f"\ndef run_eval():\n"
    f"    return f(g())")
list_puzzle_correct=[]
inf_mode="vllm"
bs = args.arg_bs_test
import time
list_puzzle_gen=[[] for _ in range(len(list_testset))]
total_speed_token=0
n_count_speed_tok=0
total_t_time=0





#Compute greedy
sampling_params = SamplingParams(n=args.arg_k,
            temperature=0.0,
            max_tokens=1024,
            # presence_penalty=1.15,
            # stop_token_ids=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
            
        )
result_greedy = llm.generate(list_testset, sampling_params)
list_puzzles_greedy_raw = []
list_puzzles_greedy =[]
for output in result_greedy:
    list_puzzles_greedy_raw.append(output.outputs[0].text)

for idx_out_gen in range(len(list_puzzles_greedy_raw)):
    txt_out=list_puzzles_greedy_raw[idx_out_gen]
    
    #post process
    try:
        txt_out=txt_out.replace("```python","```").replace("```Python","```")
        if "```" in txt_out:
            extract_g = txt_out.split("```")[1].split("assert")[0]
        else:
            if "assert" in txt_out:
                extract_g = txt_out.split("assert")[0]
        extract_g+"\nassert f(g()) == True\n"
    except:
        extract_g=txt_out

    extract_g = extract_g+"\nassert f(g()) == True\n"
    test_fg= "from typing import List\n"+ testset_f[idx_out_gen] +"\n"+extract_g + str_to_add
    # print(test_fg)
    # print("\n============\n")
    list_puzzles_greedy[idx_out_gen].append(test_fg)


list_all_puzzle_2test_greedy=[]
list_task_id_greedy=[]
for i in range(len(list_puzzle_gen)): # along the bs
    list_all_puzzle_2test_greedy.extend(list_puzzle_gen[i])
    list_task_id_greedy.extend([i for _ in range(len(list_puzzle_gen[i]))])

    
        # if j<1:
        #     print("\n-------------------\n")
        #     print(test_fg)
        
    
    # list_valid_puzzles = judge_parallel(list_puzzle_gen[i])


out_res_greedy=evaluate(list_all_puzzle_2test_greedy,list_task_id=list_task_id_greedy,entry_point="run_eval")

pass_1_greeedy = np.mean(list(out_res_greedy["pass@k"].values()))


# COMPUTE PASS@K
sampling_params = SamplingParams(n=args.arg_k,
            temperature=0.8,
            top_p=1,
            max_tokens=1024,
            # presence_penalty=1.15,
            # stop_token_ids=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
            
        )

start_time_global = time.time()
for idx in tqdm(range(curr_idx,len(list_testset),bs)): #len(dataset["test"])
    # idx=0
    print(f"\n\n============ idx {idx} ==================\n")
    flag=True
    attempt=0
    list_puzzle_idx=[]
    list_prompt=[]
    list_prompt_f=[]
    subset_test = list_testset[idx:idx+bs]
    list_prompt=subset_test

    num_tokens=0
    time_start = time.time()

    result = llm.generate(list_prompt, sampling_params)


    for idx_gen in range(num_return_sequences):
        
        generated_texts = []
        for output in result:
            num_tokens += len(output.outputs[idx_gen].token_ids)
            generated_texts.append(output.outputs[idx_gen].text)

        for idx_out_gen in range(len(generated_texts)):
            txt_out=generated_texts[idx_out_gen]
            
            #post process
            try:
                txt_out=txt_out.replace("```python","```").replace("```Python","```")
                if "```" in txt_out:
                    extract_g = txt_out.split("```")[1].split("assert")[0]
                else:
                    if "assert" in txt_out:
                        extract_g = txt_out.split("assert")[0]
                extract_g+"\nassert f(g()) == True\n"
            except:
                extract_g=txt_out

            extract_g = extract_g+"\nassert f(g()) == True\n"
            test_fg= "from typing import List\n"+ testset_f[idx+idx_out_gen] +"\n"+extract_g + str_to_add
            # print(test_fg)
            # print("\n============\n")
            list_puzzle_gen[idx+idx_out_gen].append(test_fg)

        
    time_end = time.time()
    
    print(f"speed : {num_tokens/(time_end-time_start)} token/s")
    total_speed_token+=num_tokens/(time_end-time_start)
    n_count_speed_tok+=1
    total_t_time+=time_end-time_start
    print(f"time avg over  {n_count_speed_tok} run = {time_end-time_start} ")
stop_time_global = time.time()

print(f"total time: {stop_time_global-start_time_global}")
print(f"total time: {stop_time_global-start_time_global}")

list_all_puzzle_2test=[]
list_task_id=[]
for i in range(len(list_puzzle_gen)): # along the bs
    list_all_puzzle_2test.extend(list_puzzle_gen[i])
    list_task_id.extend([i for _ in range(len(list_puzzle_gen[i]))])

    
        # if j<1:
        #     print("\n-------------------\n")
        #     print(test_fg)
        
    
    # list_valid_puzzles = judge_parallel(list_puzzle_gen[i])


out_res=evaluate(list_all_puzzle_2test,list_task_id=list_task_id,entry_point="run_eval")
# list_valid_puzzles=[res_idx["correct"] for res_idx in out_res['ordered_results']]
out_res["eval"] # {task_id_0: [res1,res2,...],task_id_1: [res1,res2,...],...}
list_all_correct_values=[]
for items,value in out_res["eval"].items():
    list_all_correct_values.append(value)

print(len(list_all_correct_values),len(list_all_correct_values[0])) # 981*k
for i in range(len(list_all_correct_values)):
    n_correct= np.sum(list_all_correct_values[i])
    n_sample=num_return_sequences

    pass_k = pass_at_k(n_sample, n_correct, k=num_return_sequences)
    list_passk.append(pass_k)
    #compute passk for k=[1,...,num_return_sequences]
    for idx_passk in range(num_return_sequences):
        pass2add=pass_at_k(n_sample, n_correct, k=idx_passk+1)
        list_all_passk[idx_passk].append(pass2add)
        # testset[idx + i][f'pass_{idx_passk+1}'] = pass2add



    # proba_solved = n_correct / n_sample
    # testset[idx + i]['proba_solved'] = float(proba_solved)
    # testset[idx + i]['n_sample'] = int(n_sample)
    # testset[idx + i]['n_correct'] = int(n_correct)
    # testset[idx + i]['generated_text'] = list_generated_text[i]
    # testset[idx + i]['parsed_puzzles'] = list_puzzle_gen[i]
    # testset[idx + i]['prompt'] = list_prompt[i]

    
print(f"correct puzzles: {int(np.sum(list_passk))}/{len(list_passk)}")
# with open(name_json+".json", "w") as outfile:
#     json.dump(list_passk,outfile)

for idx_passk in range(num_return_sequences):
    print(f"pass {idx_passk+1}: {np.sum(list_all_passk[idx_passk])}/{len(list_all_passk[idx_passk])}")
dic_passk={}
for idx_passk in range(num_return_sequences):
    dic_passk[f"pass_{idx_passk+1}"]=float(np.sum(list_all_passk[idx_passk]))

total_speed_token=total_speed_token/n_count_speed_tok
total_t_time=total_t_time/n_count_speed_tok
final_results = {
    'parameters': params,
    'results_greedy': pass_1_greeedy,#TODO
    'results': dic_passk,
    'speed': {"total_t_time":total_t_time,"total_speed_token":total_speed_token}

}
print(final_results)


n_try=0
while n_try<30:
    n_try+=1
    sleeptime = random.uniform(1, 30)
    print("sleeping for:", sleeptime, "seconds")
    sleep(sleeptime)

    try:
        with open(name_json_save_all, "r+") as outfile:
            portalocker.lock(outfile, portalocker.LOCK_EX)  # Lock the file for exclusive writing
            json_content=json.load(outfile)
            if model_id not in json_content:
                json_content[model_id]=[]
            json_content[model_id].append(final_results) 
            outfile.seek(0)
            json.dump(json_content, outfile,indent=4)
            outfile.truncate()  # Truncate file size in case new data is smaller
            portalocker.unlock(outfile)
            # n_try=30
            break


    except:
        pass

total_speed_token=total_speed_token/n_count_speed_tok
total_t_time=total_t_time/n_count_speed_tok

final_results_speed = {
    'parameters': params,
    'results': {"total_t_time":total_t_time,"total_speed_token":total_speed_token}
}
# n_try=0
# while n_try<30:
#     n_try+=1
#     sleeptime = random.uniform(1, 30)
#     print("sleeping for:", sleeptime, "seconds")
#     sleep(sleeptime)

#     try:
#         with open(name_json_save_speed, "r+") as outfile:
#             portalocker.lock(outfile, portalocker.LOCK_EX)  # Lock the file for exclusive writing
#             json_content=json.load(outfile)
#             if model_id not in json_content:
#                 json_content[model_id]=[]
#             json_content[model_id].append(final_results_speed) 
#             outfile.seek(0)
#             json.dump(json_content, outfile,indent=4)
#             outfile.truncate()  # Truncate file size in case new data is smaller
#             portalocker.unlock(outfile)
#             n_try=30
#             break


#     except:
#         pass

list_res={}
for task_id, task_results in out_res["eval"].items():
    list_res[task_id]=task_results
n_try=0
while n_try<30:
    n_try+=1
    sleeptime = random.uniform(1, 30)
    print("sleeping for:", sleeptime, "seconds")
    sleep(sleeptime)

    try:
        with open(name_json_save_all_solution, "r+") as outfile:
            portalocker.lock(outfile, portalocker.LOCK_EX)  # Lock the file for exclusive writing
            json_content=json.load(outfile)
            # if args.name_archive not in json_content:
            #     json_content[args.name_archive]=[]
            json_content[args.name_archive]=list_res 
            outfile.seek(0)
            json.dump(json_content, outfile,indent=4)
            outfile.truncate()  # Truncate file size in case new data is smaller
            portalocker.unlock(outfile)
            n_try=30
            break


    except:
        pass







print(f"total time: {stop_time_global-start_time_global}")
print(f"total time: {stop_time_global-start_time_global}")
# with open(name_json_save_all, "r") as outfile:
#     json_content=json.load(outfile)
# json_content[run_name]=dic_passk 
# with open(name_json_save_all, "w") as outfile:
#     json.dump(json_content,outfile,indent=4)


# with open(name_json+"_e"+str(num_train_epochs)+"_seed_"+seed+".json", "w") as outfile:
#     json.dump(json_content,outfile,indent=4)
# with open(name_json_sol+"_e"+str(num_train_epochs)+"_seed_"+seed+".json", "w") as outfile:
#     json.dump(list_all_puzzle,outfile,indent=4)

# with open(filename_save, 'w') as f:
#     json.dump(final_results, f, indent=4)




