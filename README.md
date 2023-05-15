ChatGPT解决Huggingface上CodeXGLUE数据集的任务的通用框架


```
python main.py --result_path code --dataset code_x_glue_cc_code_refinement --subset medium medium --subset_index 1 --subset_all --split test --key_input buggy --key_output fixed --trunc_size 2000

python main.py --result_path code --dataset code_x_glue_cc_code_refinement --subset medium medium --subset_index 1 --subset_all --split test --key_input buggy --key_output fixed --trunc_size 2000

python evaluation\acc\evaluator.py code_repair\medium\eval\medium_pred.txt code_repair\medium\eval\medium_true.txt > code_repair\medium\eval\medium_acc.txt

python evaluation\bleu\evaluator.py code_repair_ape\medium\eval\medium_pred.txt code_repair_ape\medium\eval\medium_true.txt > code_repair_ape\medium\eval\medium_bleu.txt

cd evaluation\CodeBLEU
python calc_code_bleu.py --refs code_repair\medium\eval\medium_pred.txt --hyp code_repair\medium\eval\medium_true.txt --lang java

```