# Copyright (c) Microsoft Corporation. 
# Licensed under the MIT license.

from __future__ import absolute_import
import os
import sys
import codecs
from bleu import _bleu
import numpy as np
import tiktoken
import pickle

def tokens_from_string(string: str, model_name: str = 'gpt-3.5-turbo-0301') -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.encoding_for_model(model_name)
    tokens = encoding.encode(string)
    return tokens

def cal_bleu(hyp, ref):
    dev_bleu = round(_bleu(ref, hyp), 2)
    print ("bleu-4: ", str(dev_bleu))

def tokens_from_file(path):
    with open(path,'r') as f:
        lines = f.readlines()
    tokens = [tokens_from_string(line) for line in lines]
    return tokens

if __name__ == "__main__":
    hyp_path = sys.argv[1] # pred
    ref_path = sys.argv[2] # gold
    # hyp_path = r'E:\Courses\NLP\openai\general\code_repair\small\eval\small_pred.txt'
    # ref_path = r'E:\Courses\NLP\openai\general\code_repair\small\eval\small_true.txt'
    hyp = tokens_from_file(hyp_path)
    ref = tokens_from_file(ref_path)
    ref = [[_] for _ in ref]
    cal_bleu(hyp, ref)

