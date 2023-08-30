# unit-test-code-gen

## Finetune llm to generate unit test code

To execute the fine-tuning script run the following command:

```
python finetune/finetune.py \
  --model_path="codellama/CodeLlama-7b-hf"\
  --dataset_name="jitx/Methods2Test_java_unit_test_code"\
  --split="train"\
  --size_valid_set 10000\
  --streaming\
  --seq_length 2048\
  --max_steps 1000\
  --batch_size 1\
  --input_column_name="src_fm"\
  --output_column_name="target"\ 
  --gradient_accumulation_steps 16\
  --learning_rate 1e-4\
  --lr_scheduler_type="cosine"\
  --num_warmup_steps 100\
  --weight_decay 0.05\
  --output_dir="./checkpoints" \
```