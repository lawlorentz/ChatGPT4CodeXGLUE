# Copyright (c) Microsoft Corporation. 
# Licensed under the MIT license.

from __future__ import absolute_import
import os
import sys
import codecs
import numpy as np
import tiktoken
import pickle

def tokens_from_string(string):
    tokens = string.replace(' \n','').replace('\n','')
    tokens = tokens.split(' ')
    return tokens


def tokens_from_file(path):
    with open(path,'r') as f:
        lines = f.readlines()
    lines = [line.replace(' \n', '') for line in lines]
    lines = [line.replace('\n', '') for line in lines]
    tokens = [tokens_from_string(line) for line in lines]
    return tokens

def strs_from_file(path):
    with open(path,'r') as f:
        lines = f.readlines()
    lines = [line.replace(' \n', '') for line in lines]
    lines = [line.replace('\n', '') for line in lines]
    return lines

if __name__ == "__main__":
    hyp_path = sys.argv[1] # pred
    ref_path = sys.argv[2] # gold
    # hyp_path = r'E:\Courses\NLP\openai\general\code_repair_ape\medium\eval\medium_pred.txt'
    # ref_path = r'E:\Courses\NLP\openai\general\code_repair_ape\medium\eval\medium_true.txt'
    hyp = strs_from_file(hyp_path)
    ref = strs_from_file(ref_path)
    acc = 0
    assert len(hyp)==len(ref)
    for i in range(len(hyp)):
        a = hyp[i]
        b = ref[i]
        if a==b:
            acc+=1
            # print(1)
    acc/=len(ref)
    print(acc)
