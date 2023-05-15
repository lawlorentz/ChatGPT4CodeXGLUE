#!/bin/bash

# 参数定义
result_path="code"
dataset="code_x_glue_cc_code_refinement"
subset=("small" "medium")
subset_index=0
split="test"
key_input="buggy"
key_output="fixed"
trunc_size=2000

python main.py \
  --result_path "$result_path" \
  --dataset "$dataset" \
  --subset "${subset[@]}" \
  --subset_index "$subset_index" \
  --split "$split" \
  --key_input "$key_input" \
  --key_output "$key_output" \
  --trunc_size "$trunc_size"

# for ((subset_index_=0; subset_index_<${#subset[@]}; subset_index_++)); do
#   python json2txt.py \
#     --result_path "$result_path" \
#     --subset "${subset[@]}" \
#     --subset_index "$subset_index_"
# done

python json2txt.py \
  --result_path "$result_path" \
  --subset "${subset[@]}" \
  --subset_index "$subset_index"

