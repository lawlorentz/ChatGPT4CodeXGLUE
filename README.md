ChatGPT解决Huggingface上CodeXGLUE数据集的任务的通用框架


```
python main.py --result_path code --dataset code_x_glue_cc_code_refinement --subset small medium --subset_index 1 --subset_all --split test --key_input buggy --key_output fixed --trunc_size 2000

python evaluation\acc\evaluator.py E:\Courses\NLP\openai\general\code_repair_my_prompt\small\eval\small_pred.txt E:\Courses\NLP\openai\general\code_repair_my_prompt\small\eval\small_true.txt > E:\Courses\NLP\openai\general\code_repair_my_prompt\small\eval\small_acc.txt

python evaluation\bleu\evaluator.py E:\Courses\NLP\openai\general\code_repair_ape\small\eval\small_pred.txt E:\Courses\NLP\openai\general\code_repair_ape\small\eval\small_true.txt > E:\Courses\NLP\openai\general\code_repair_ape\small\eval\small_bleu.txt

cd evaluation\CodeBLEU
python calc_code_bleu.py --refs E:\Courses\NLP\openai\general\code_repair\small\eval\small_pred.txt --hyp E:\Courses\NLP\openai\general\code_repair\small\eval\small_true.txt --lang java

```