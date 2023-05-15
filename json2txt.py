from __future__ import absolute_import
import tiktoken
import json
import html
import pickle
import os
import sys
import numpy as np
import re


def tokens_from_string(string: str, model_name: str = 'gpt-3.5-turbo-0301') -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.encoding_for_model(model_name)
    encode_tokens = encoding.encode(string)
    tokens = [encoding.decode_single_token_bytes(
        token) for token in encode_tokens]
    return tokens

def tokenize_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    text = data['response']["choices"][0]["message"]["content"]

    def process(t):
        t = html.unescape(t.lower())
        # t = re.sub('<[^>]*>', '', t)
        # t = t.replace('/n', '')
        # t = t.replace(' ','')
        return t
    text = process(text)
    gold = process(data['ground truth'])
    return tokens_from_string(text), tokens_from_string(gold)


if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Process some arguments')
    parser.add_argument('--result_path', type=str, default='code_repair_ape', help='Path to save the result')
    parser.add_argument('--subset', type=str, nargs='+', default=['small', 'medium'], help='Subset(s) of the dataset')
    parser.add_argument('--subset_index', type=int, default=0, help='Index of the subset to use')
    config = parser.parse_args()

    subset = config.subset
    subset_index = config.subset_index
    result_root_path = config.result_path
    result_path = os.path.join(result_root_path, subset[subset_index])

    checkpoint_path = os.path.join(
        result_path, f'checkpoint_{subset[subset_index]}.json')
    with open(checkpoint_path, 'r') as f:
        data = json.load(f)
        lenth = data["num"] + 1


    texts = [[] for _ in range(lenth)]
    golds = []
    s = subset[subset_index]
    for i in range(lenth):
        path = os.path.join(result_path, f'{s}_test_{i}.json')
        text, gold = tokenize_json(path)
        texts[i].append(text)
        golds.append(gold)
    save_path = os.path.join(result_path, f'eval')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_file_path1 = os.path.join(save_path, f'{s}_pred.txt')
    save_file_path2 = os.path.join(save_path, f'{s}_true.txt')
    with open(save_file_path1, 'w') as f:
        for text in texts:
            f.write(b''.join(text[0]).decode('utf-8').replace('\n',''))
            f.write('\n')
    with open(save_file_path2, 'w') as f:
        for gold in golds:
            f.write(b''.join(gold).decode('utf-8'))
    