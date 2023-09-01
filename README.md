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
  --load_in_8bit\
  --size_valid_set 10000\
  --streaming\
  --seq_length 2048\
  --max_steps 10000\
  --batch_size 8\
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

## How GPU is utilized during traning
- [How GPU is utilized during training](https://huggingface.co/docs/transformers/model_memory_anatomy)
  - Tensor Contractions, Linear layers and components of Multi-Head Attention all do batched matrix-matrix multiplications. These operations are the most compute-intensive part of training a transformer.
    - High Arithmetic Intensity: Matrix-matrix multiplications involve a large number of arithmetic operations (multiplications and additions) for each element in the resulting matrix. This high arithmetic intensity means that a significant amount of computation must be performed for each element of the output matrix.

    - Memory Access Patterns: Matrix-matrix multiplications require access to a large amount of data from memory. The input matrices (often referred to as A and B matrices) need to be read from memory, and the resulting matrix (C matrix) needs to be written back to memory. These memory access patterns can result in significant memory bandwidth requirements, and if memory access is not optimized, it can lead to performance bottlenecks.

    - Parallelization Challenges: While matrix-matrix multiplications are inherently parallelizable, efficiently parallelizing them across multiple processor cores or GPU threads can be challenging. Coordinating the parallel execution of a large number of multiplications and additions while minimizing communication between processing units can be complex.
  - Statistical Normalizations, Softmax and layer normalization are less compute-intensive than tensor contractions, and involve one or more reduction operations, the result of which is then applied via a map.
  - Element-wise Operators, These are the remaining operators: biases, dropout, activations, and residual connections. These are the least compute-intensive operations.
- [Data Movement Is All You Need: A Case Study on Optimizing Transformers 2020](https://arxiv.org/pdf/2007.00072.pdf)
- Anatomy of Model's Memory

    We’ve seen that training the model uses much more memory than just putting the model on the GPU. This is because there are many components during training that use GPU memory. The components on GPU memory are the following:

      1. model weights
      2. optimizer states
      3. gradients
      4. forward activations saved for gradient computation
      5. temporary buffers
      6. functionality-specific memory

    A typical model trained in mixed precision with AdamW requires 18 bytes per model parameter plus activation memory. For inference there are no optimizer states and gradients, so we can subtract those. And thus we end up with 6 bytes per model parameter for mixed precision inference, plus activation memory.

    Let’s look at the details.

    **Model Weights:**

      - 4 bytes * number of parameters for fp32 training
      - 6 bytes * number of parameters for mixed precision training. In mixed precision training, you typically maintain two copies of the model in memory: one in 32-bit floating-point format (FP32) and another in 16-bit floating-point format (FP16)

    **Optimizer States:**

      - 8 bytes * number of parameters for normal AdamW (maintains 2 states)
      - 2 bytes * number of parameters for 8-bit AdamW optimizers like `bitsandbytes`
      - 4 bytes * number of parameters for optimizers like SGD with momentum (maintains only 1 state)

    **Gradients**

      - 4 bytes * number of parameters for either fp32 or mixed precision training (gradients are always kept in fp32)
    
    **Forward Activations**

      - size depends on many factors, the key ones being sequence length, hidden size and batch size.
    There are the input and output that are being passed and returned by the forward and the backward functions and the forward activations saved for gradient computation.


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