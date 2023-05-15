import os
import sys
from datasets import load_dataset
import tiktoken
import re
import json
import openai
from tqdm import tqdm
import time
import apikey

keys = apikey.apikeys

os.environ["http_proxy"] = "http://127.0.0.1:7890"
os.environ["https_proxy"] = "https://127.0.0.1:7890"


def num_tokens_from_string(string: str, model_name: str = 'gpt-3.5-turbo-0301') -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.encoding_for_model(model_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def truncate_string(string: str, lenth, model_name: str = 'gpt-3.5-turbo-0301') -> str:
    encoding = tiktoken.encoding_for_model(model_name)
    trunc_string = encoding.encode(string)[:lenth]
    return encoding.decode(trunc_string)


def restart_if_failed(func, max_tries, args=(), kwargs={}, sleep=None):
    '''
    re-run when some exception happens, until `max_tries`
    '''
    import traceback
    from collections import deque

    dq = deque(maxlen=max_tries)
    while True:
        dq.append(time.time())
        try:
            return func(*args, **kwargs)
        except Exception:
            traceback.print_exc()
            if len(dq) >= max_tries:
                break
            if sleep is not None:
                time.sleep(sleep)
        else:
            break


def main(config):
    result_root_path = config.result_path
    dataset = config.dataset
    subset = config.subset
    subset_index = config.subset_index
    split = config.split
    key_input = config.key_input
    key_output = config.key_output
    trunc_size = config.trunc_size
    prompt_template = config.prompt_template
    instruction = config.instruction

    result_path = os.path.join(result_root_path, subset[subset_index])
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    dataset = load_dataset(dataset, subset[subset_index])

    prompt0 = prompt_template.replace('{Instruction}', instruction)

    with open(os.path.join(result_root_path,'prompt.txt'),'w') as f:
        f.write(prompt0)

    start_idx = 0
    checkpoint_path = os.path.join(
        result_path, f'checkpoint_{subset[subset_index]}.json')
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'r') as f:
            data = json.load(f)
            start_idx = data["num"] + 1

    for index in tqdm(range(start_idx, len(dataset[split]))):
        openai.api_key = keys[index%(len(keys))]

        input_ = dataset[split][index][key_input]
        prompt = prompt0.replace('{Input}', input_)

        num_tokens = num_tokens_from_string(prompt)
        if num_tokens > trunc_size:
            # print(num_tokens)
            prompt = truncate_string(prompt, trunc_size)

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0301",
            temperature=0,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
                # {"role": "assistant", "content": ""},
            ]
        )

        # content = response["choices"][0]["message"]["content"]
        # print(content)

        save_json = {'split': split,
                     'index': index,
                     'prompt': prompt,
                     'response': response,
                     'ground truth': dataset[split][index][key_output],
                     'truncate': num_tokens > trunc_size
                     }

        with open(os.path.join(result_path, f'{subset[subset_index]}_{split}_{index}.json'), 'w') as f:
            json.dump(save_json, f)

        with open(checkpoint_path, 'w') as f:
            json.dump({'num': index}, f)
        # time.sleep(20/len(keys)-1)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Process some arguments')
    parser.add_argument('--result_path', type=str, default='code_repair_ape', help='Path to save the result')
    parser.add_argument('--dataset', type=str, default='code_x_glue_cc_code_refinement', help='Name of the dataset')
    parser.add_argument('--subset', type=str, nargs='+', default=['small', 'medium'], help='Subset(s) of the dataset')
    parser.add_argument('--subset_index', type=int, default=0, help='Index of the subset to use')
    parser.add_argument('--subset_all', type=bool, default=False, const=True, nargs='?', help='whether to uses all of the subsets')
    parser.add_argument('--split', type=str, default='test', help='Split of the dataset to use')
    parser.add_argument('--key_input', type=str, default='buggy', help='Key for the input data')
    parser.add_argument('--key_output', type=str, default='fixed', help='Key for the output data')
    parser.add_argument('--trunc_size', type=int, default=2000, help='Size to truncate the input sequence')
    parser.add_argument('--prompt_template', type=str, default='[Instruction]\n{Instruction}\n[Input code]\n{Input}\n[Output code]\n', help='prompt_template')
    parser.add_argument('--instruction_path', type=str, default='./instruction.txt', help='instruction in the prompt')
    parser.add_argument('--instruction', type=str, default='', help='instruction in the prompt')
    config = parser.parse_args()
    
    with open(config.instruction_path,'r') as f:
        config.instruction = f.read()

    if config.subset_all:
        for i in range(len(config.subset)):
            config.subset_index = i
            restart_if_failed(main, 1000, args=(config,))
    else:
        restart_if_failed(main, 1000, args=(config,))

