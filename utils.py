import numpy as np
from pebble import ProcessPool
import ast
import copy
# import tiktoken
import json
import re
import os
import numpy as np
from typing import List


from prompt import get_prompt_template
def create_conversation(sample,retun_response=True):

    system_message = """You are helpfull assistant."""
    instruction,response = get_prompt_template(sample,retun_response)
    if retun_response:
        return {
            "chat": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": instruction},
                {"role": "assistant", "content": response}
                ]
            }
    else:
        return {
            "chat": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": instruction}
                ]
            }
def remove_keys(dataset,keys=["emb","target_skills","puzzle_history","is_valid","is_valid_explanation"]):
    for i in dataset:
        for j in keys:
            if j in i:
                del i[j]
    return dataset

def get_formated_chat_dataset(dataset,text_field="program_str",retun_response=True,return_hf_data=True):
    
    new_dataset=[]
    for i in range(len(dataset)):
        data_i= dataset[i][text_field]
        data_i_formated=create_conversation(data_i,retun_response=retun_response)
        new_dataset.append(data_i_formated)
    if return_hf_data:
        from datasets import Dataset
        return Dataset.from_list(new_dataset).shuffle(seed=42)
    else:
        return new_dataset
      



#test P3

def pass_at_k(n, c, k):
    """
    Adapted from "Evaluating Large Language Models Trained on Code" (https://arxiv.org/abs/2107.03374)

    :param n: total number of samples
    :param c: number of correct samples
    :param k: k in pass@k
    """
    assert n >= k
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))


def test_puzzle(test_fg):
    test_fg= "from typing import *\n"+test_fg
    try:
        exec(test_fg)
        return True,test_fg
    except Exception as e:
        # print(str(e))
        # print("program not working: "+test_fg)
        return False,test_fg
import multiprocessing
def judge_parallel(src_codes, timeout=3., max_workers=multiprocessing.cpu_count()-1):

    max_workers = min(len(src_codes), max_workers)

    codes = src_codes
    successes = set()
    with ProcessPool(max_workers=max_workers) as pool:
        future = pool.map(test_puzzle, [code for code in codes], timeout=timeout)

        results = future.result()
        i = 0
        while True:
            try:
                success, code = next(results)
                if success:
                    successes.add(codes[i])
            except StopIteration:
                break
            except (TimeoutError, Exception) as error:
                pass
            assert i < len(codes)
            i += 1
        assert i == len(codes)
    # utils.silence_std_err(False)
    return [code in successes for code in src_codes]



def extract_arguments_except_first_specific(func_code, function_name='f'):
    # Parse the source code into an AST
    tree = ast.parse(func_code)
    
    # Initialize the result string
    result = []
    
    # Visit each node in the AST
    for node in ast.walk(tree):
        # Check if the node is a function definition and matches the specified function name
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            # Get the arguments from the function definition
            args = node.args
            
            # Exclude the first positional argument
            pos_args = args.args[1:]  # Skip the first argument
            
            # Handle positional arguments with defaults
            defaults = args.defaults
            num_defaults = len(defaults)
            num_pos_args = len(pos_args)
            default_start_index = num_pos_args - num_defaults

            # Handle non-default arguments
            for i, arg in enumerate(pos_args):
                if i >= default_start_index:
                    # If the argument has a default value, include it
                    default_value = defaults[i - default_start_index]
                    result.append(f"{ast.unparse(arg)}={ast.unparse(default_value)}")
                else:
                    # If no default, just add the argument
                    result.append(ast.unparse(arg))
            
            # Include *args and **kwargs
            if args.vararg:
                result.append(ast.unparse(args.vararg))
            if args.kwarg:
                result.append(ast.unparse(args.kwarg))

            # Handle keyword-only arguments with defaults
            for kw, kw_default in zip(args.kwonlyargs, args.kw_defaults):
                if kw_default is None:
                    result.append(ast.unparse(kw))
                else:
                    result.append(f"{ast.unparse(kw)}={ast.unparse(kw_default)}")
            break  # Stop if the target function is found
    
    return ', '.join(result)
