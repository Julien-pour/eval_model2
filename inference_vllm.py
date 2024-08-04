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
parser.add_argument("-m", "--arg_model_id", type=str, help=" model",default="deepseek-coder-1.3b-instruct")
parser.add_argument("-e", "--arg_epoch", type=int, help="number epoch",default=2)
parser.add_argument("-b", "--arg_bs", type=int, help=" bs train",default=2)
parser.add_argument( "--arg_bs_test", type=int, help=" bs train",default=1024)

parser.add_argument("-s", "--arg_seed_random", type=int, help="seed for the random generator",default=1)
parser.add_argument("-g", "--arg_gpu", type=str, help="GPU use",default="a100")
parser.add_argument("-a", "--accum_step", type=int, help="number of accumulation step",default=1)
parser.add_argument("--test_base_model", type=str, help="just test base model",default="False")
parser.add_argument("--lr", type=float, help="learning rate")
parser.add_argument("--name_run", type=str, help="run_name")
parser.add_argument("-k", "--arg_k", type=int, help="k in pass@k",default=1)
parser.add_argument("--n_gpu", type=int, help="how many gpu to use",default=2)
parser.add_argument("--eager_mode", type=str, help="eager_mode",default="False")
parser.add_argument("--swap_space", type=float, help="swap space",default=1)
parser.add_argument("--seed", type=int, help="seed: -1 -> merged, ...",default=-1) 
parser.add_argument("--path_archive", type=str, help="name_archive, ... archives/") 
parser.add_argument("--path_archive_test", type=str, help="name_archive, ... archives/",default="archives/P3_test_emb_wizard3B.json") 

parser.add_argument("--temperature", type=float, help="temperature",default=0.0) 
parser.add_argument("--file_save_name", type=str, help="file_save_name",default="passk_rebuttal") 
parser.add_argument("--cutoff_gen", type=int, help="gen to keep -1 for all gen",default=-1) 



n_max_token=2048 #1360*

args = parser.parse_args()
eager_mode=args.eager_mode.lower()=="true"
learning_rate= args.lr
model_id =  args.arg_model_id

accum_step=args.accum_step

# name: name of the methode (aces,elm-nlp,aces)
unique_id=args.path_archive.split("/")[-1].split(".")[0]
name_archive_test=args.path_archive_test.split("/")[-1].split(".")[0]

params={"lr":args.lr,"epochs":args.arg_epoch,"model_id":model_id,"test_base_model":args.test_base_model,
        "name_archive":unique_id,"name_archive_test":name_archive_test,"seed":args.seed,"accum_step":accum_step,
        "gpu":args.arg_gpu,"n_gpu":args.n_gpu, "temperature":args.temperature }

# try:
#     unique_id=f"{os.getenv('SLURM_ARRAY_JOB_ID')}_{os.getenv('SLURM_ARRAY_TASK_ID')}"
# except:
#     unique_id=f"{os.getenv('SLURM_ARRAY_JOB_ID')}_0"
    

filename_save = args.base_path+"save_results/multiple_results/"+f"good_results_{unique_id}.json"

params["unique_id"]=unique_id

run_name = model_id+"_"+unique_id

if args.arg_gpu=="a100":
    name_json_save_all = args.base_path+f"save_results/{args.file_save_name}_a100.json"#.split("/")[1]
else:
    name_json_save_all = args.base_path+f"save_results/{args.file_save_name}.json"#.split("/")[1]

name_json_save_all_solution = args.base_path+f"save_results/save_sol/good_{model_id}.json"#.split("/")[1]


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



hf_dir=args.path_model_base


output_dir = hf_dir+run_name


if args.test_base_model.lower()=="true":
    output_dir = hf_dir+model_id
if args.arg_gpu=="v100" or "gptq" in model_id.lower():
    dtype="half"
else:
    dtype="auto"
if "Mixtral" in model_id or "Qwen" in model_id or "starcoder" in model_id:
    eager_mode=True
else:
    eager_mode=False
from transformers import AutoTokenizer
tokenizer=AutoTokenizer.from_pretrained(output_dir)

print("load model at", output_dir)
llm = LLM(output_dir,max_model_len=4000,enforce_eager=eager_mode,tensor_parallel_size=args.n_gpu,dtype=dtype,swap_space=args.swap_space)
sampling_params = SamplingParams(
            temperature=args.temperature,
            top_p=1,
            max_tokens=1024,
            # presence_penalty=1.15,
            stop_token_ids=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
            
        )


# testset= preprocessing_P3_no_test(split="test",n_token_max=1024,path=args.base_path,tokenizer=tokenizer)
path_test=args.base_path+args.path_archive_test
with open(path_test,mode = "r") as f:
    testset = json.load(f)


if args.cutoff_gen!=-1:
    testset =[p for p in testset if p["idx_generation"] <= args.cutoff_gen ]
# testset = [x for x in testset if (x["idx_generation"]<=39 and x["idx_generation"]>=0)]

testset_f=[p["program_str"].split("def g")[0].strip() for p in testset]
curr_idx=0
correct_puzz=0

num_return_sequences=args.arg_k #n_try
list_all_passk=[[] for i in range(num_return_sequences)]
list_passk=[]

list_puzzle=[]

    
# list_testset= [x["program_str"] for x in testset]
dat_chat = get_formated_chat_dataset(testset,text_field="program_str",retun_response=False,return_hf_data=False)
list_testset= []


template_codegemma="""<bos><start_of_turn>user
{instruction}<end_of_turn>
<start_of_turn>model
""" # idk why it is not working with chat template

for chat in dat_chat:
    for i in range(len(chat["chat"])):
        idx_to_del=[]
        if "Mistral-7B" in model_id or "mixtral" in model_id.lower() or "codegem" in model_id or "starcod" in model_id.lower():
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

    for idx_gen in range(num_return_sequences):
        num_tokens=0
        time_start = time.time()

        result = llm.generate(list_prompt, sampling_params)
        
        generated_texts = []
        for output in result:
            num_tokens += len(output.outputs[0].token_ids)
            generated_texts.append(output.outputs[0].text)

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
    'results': dic_passk,
    'speed': {"total_t_time":total_t_time,"total_speed_token":total_speed_token}

}
print(final_results)


n_try=0
while n_try<30:
    n_try+=1
    sleeptime = random.uniform(1, 10)
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

# final_results_speed = {
#     'parameters': params,
#     'results': {"total_t_time":total_t_time,"total_speed_token":total_speed_token}
# }
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

# with open(name_json_save_all_solution,"w") as outfile:
#     json.dump(list_puzzle_gen,outfile,indent=4)



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




