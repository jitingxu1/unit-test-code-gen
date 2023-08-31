# unit-test-code-gen

## Java unit test dataset
[Methods2Test_java_unit_test_code](https://huggingface.co/datasets/jitx/Methods2Test_java_unit_test_code)

## codellama

To use this model, please make sure to install transformers from main until the next version is released:
```
pip install git+https://github.com/huggingface/transformers.git@main
```
## Finetune codellamma-7b to generate Java Junit test code

To execute the fine-tuning script run the following command:

```
python src/finetune/finetune.py \
  --model_path="codellama/CodeLlama-7b-hf"\
  --dataset_name="jitx/Methods2Test_java_unit_test_code"\
  --split="train"\
  --size_valid_set 10000\
  --streaming\
  --seq_length 1024\
  --max_steps 10000\
  --batch_size 1\
  --input_column_name="src_fm"\
  --output_column_name="target"\ 
  --gradient_accumulation_steps 16\
  --learning_rate 1e-4\
  --lr_scheduler_type="cosine"\
  --num_warmup_steps 100\
  --weight_decay 0.05\
  --wandb_project="finetune-codellama"\
  --wandb_watch="all"\
  --wandb_run_name="finetune-codellama"\
  --output_dir="./checkpoints" \
```

## Use wandb
The W&B integration adds rich, flexible experiment tracking and model versioning
 to interactive centralized dashboards without compromising that ease of use.

 1. You need a wandb free [account](https://wandb.ai/site?_gl=1*1m8qtk9*_ga*MTE2NTY0MDUwOS4xNjg3ODkzOTc2*_ga_JH1SJHJQXJ*MTY5MzUwOTE2My4yOS4wLjE2OTM1MDkxNjMuNjAuMC4w)
 2. setup wandb parameters
  - wandb_project: Give your project a name (huggingface by default)
  - wandb_watch: Set whether you'd like to log your models gradients, parameters or neither
    - `false` (default): No gradient or parameter logging
    - `gradients`: Log histograms of the gradients
    - `all`: Log histograms of gradients and parameters
  - wandb_run_name
  - wand_log_model: Log the model as artifact at the end of training (false by default)

## Transformer Training Arguments: [source](https://github.com/huggingface/transformers/blob/main/src/transformers/training_args.py)
- evaluation_strategy (`str` or [`~trainer_utils.IntervalStrategy`], *optional*, defaults to `"no"`):
The evaluation strategy to adopt during training. Possible values are:
  - `"no"`: No evaluation is done during training.
  - `"steps"`: Evaluation is done (and logged) every `eval_steps`.
  - `"epoch"`: Evaluation is done at the end of each epoch.
- gradient_accumulation_steps (`int`, *optional*, defaults to 1):
            Number of updates steps to accumulate the gradients for, before performing a backward/update pass.

            When using gradient accumulation, one step is counted as one step with backward pass. Therefore, logging,
            evaluation, save will be conducted every `gradient_accumulation_steps * xxx_step` training examples.
- eval_delay (`float`, *optional*):
            Number of epochs or steps to wait for before the first evaluation can be performed, depending on the
            evaluation_strategy.
- learning_rate (`float`, *optional*, defaults to 5e-5):
            The initial learning rate for [`AdamW`] optimizer.
- num_train_epochs(`float`, *optional*, defaults to 3.0):
            Total number of training epochs to perform (if not an integer, will perform the decimal part percents of
            the last epoch before stopping training).
- max_steps (`int`, *optional*, defaults to -1):
            If set to a positive number, the total number of training steps to perform. Overrides `num_train_epochs`.
            In case of using a finite iterable dataset the training may stop before reaching the set number of steps
            when all data is exhausted