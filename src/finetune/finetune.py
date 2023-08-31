import os
import glob
from dataclasses import dataclass, field
from typing import List

import peft
import torch
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
)
from datasets import Dataset, load_dataset
from torch.utils.data import IterableDataset

def prepare_sample_text(example, input_column_name="prompt", output_column_name="completion"):
    """Prepare the text from a sample of the dataset."""
    text = f"focal method: {example[input_column_name]}\n\nJUnit test: {example[output_column_name]}"
    return text

class ConstantLengthDataset(IterableDataset):
    """
    Iterable dataset that returns constant length chunks of tokens from stream of text files.
        Args:
            tokenizer (Tokenizer): The processor used for proccessing the data.
            dataset (dataset.Dataset): Dataset with text files.
            infinite (bool): If True the iterator is reset after dataset reaches end else stops.
            seq_length (int): Length of token sequences to return.
            num_of_sequences (int): Number of token sequences to keep in buffer.
            chars_per_token (int): Number of characters per token used to estimate number of tokens in text buffer.
    """

    def __init__(
        self,
        tokenizer,
        dataset,
        infinite=False,
        seq_length=1024,
        num_of_sequences=1024,
        chars_per_token=3.6,
        input_column_name="src_fm",
        output_column_name="target"
    ):
        self.tokenizer = tokenizer
        self.concat_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else args.eos_token_id
        self.dataset = dataset
        self.seq_length = seq_length
        self.infinite = infinite
        self.current_size = 0
        self.max_buffer_size = seq_length * chars_per_token * num_of_sequences
        self.input_column_name = input_column_name
        self.output_column_name = output_column_name

    def __iter__(self):
        iterator = iter(self.dataset)
        more_examples = True
        while more_examples:
            buffer, buffer_len = [], 0
            while True:
                if buffer_len >= self.max_buffer_size:
                    break
                try:
                    buffer.append(prepare_sample_text(next(iterator), self.input_column_name, self.output_column_name))
                    buffer_len += len(buffer[-1])
                except StopIteration:
                    if self.infinite:
                        iterator = iter(self.dataset)
                    else:
                        more_examples = False
                        break
            tokenized_inputs = self.tokenizer(buffer, truncation=False)["input_ids"]
            all_token_ids = []
            for tokenized_input in tokenized_inputs:
                all_token_ids.extend(tokenized_input + [self.concat_token_id])
            for i in range(0, len(all_token_ids), self.seq_length):
                input_ids = all_token_ids[i : i + self.seq_length]
                if len(input_ids) == self.seq_length:
                    self.current_size += 1
                    yield {
                        "input_ids": torch.LongTensor(input_ids),
                        "labels": torch.LongTensor(input_ids),
                    }

def chars_token_ratio(dataset, tokenizer, input_column_name="prompt", output_column_name="completion", nb_examples=400):
    """
    Estimate the average number of characters per token in the dataset.
    """
    total_characters, total_tokens = 0, 0
    for _, example in tqdm(zip(range(nb_examples), iter(dataset)), total=nb_examples):
        text = prepare_sample_text(example, input_column_name, output_column_name)
        total_characters += len(text)
        if tokenizer.is_fast:
            total_tokens += len(tokenizer(text).tokens())
        else:
            total_tokens += len(tokenizer.tokenize(text))

    return total_characters / total_tokens

def create_datasets(tokenizer, args):
    dataset = load_dataset(
        args.dataset_name,
        split=args.split,
        streaming=args.streaming,
    )
    if args.streaming:
        print("Loading the dataset in streaming mode")
        valid_data = dataset.take(args.size_valid_set)
        train_data = dataset.skip(args.size_valid_set)
        train_data = train_data.shuffle(buffer_size=args.shuffle_buffer, seed=args.seed)
    else:
        train_data = dataset["train"]
        valid_data = dataset["test"]
        print(f"Size of the train set: {len(train_data)}. Size of the validation set: {len(valid_data)}")
    
    chars_per_token = chars_token_ratio(train_data, tokenizer, args.input_column_name, args.output_column_name)
    print(f"The character to token ratio of the dataset is: {chars_per_token:.2f}")
    
    train_dataset = ConstantLengthDataset(
        tokenizer,
        train_data,
        infinite=True,
        seq_length=args.seq_length,
        chars_per_token=chars_per_token,
        input_column_name=args.input_column_name,
        output_column_name=args.output_column_name
    )
    valid_dataset = ConstantLengthDataset(
        tokenizer,
        valid_data,
        infinite=False,
        seq_length=args.seq_length,
        chars_per_token=chars_per_token,
        input_column_name=args.input_column_name,
        output_column_name=args.output_column_name
    )
    return train_dataset, valid_dataset

@dataclass
class TrainLoraArguments:
    data_path: str = field(
      default="codellama/CodeLlama-7b-hf",
      metadata={"help": "Dataset dir for training / eval "}
    )
    output_dir: str = field(
      default="./checkpoints",
      metadata={"help": "Output dir for checkpoint"}
    )
    base_model: str = field(
        default="codellama/CodeLlama-7b-hf",
        metadata={"help": "Base model for fine-tuning"}
    )

    split: str = "train"
    size_valid_set: int = 1000
    streaming: bool = True
    seq_length: int = 2048
    max_steps: int = 10000
    input_column_name: str = "src_fm"
    output_column_name: str = "target"


    batch_size: int = 128
    micro_batch_size: int = 4
    gradient_accumulation_steps: int = 16
    num_epochs: int = 3
    learning_rate: float = 3e-4
    cutoff_len: int = 256
    lr_scheduler_type: str = "cosine"
    num_warmup_steps: int = 100
    weight_decay: float = 0.005

    # Evaluations
    val_set_size: int = 2000
    eval_steps: int = 200

    # Lora Hyperparams
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = (
        [
            "q_proj",
            "v_proj",
        ],
    )
    resume_from_checkpoint: str = None  # either training checkpoint or final adapter
    half: bool = True

    #wandb
    wandb_project: str = ""
    wandb_run_name: str = ""
    wandb_watch: str = "" # options: false | gradients | all
    wandb_log_model: str = "end",  # options: false | true



def parse_args() -> TrainLoraArguments:
    parser = HfArgumentParser(TrainLoraArguments)
    return parser.parse_args()

def train(args):
    # gradient_accumulation_steps = args.batch_size // args.micro_batch_size
    gradient_accumulation_steps = args.gradient_accumulation_steps
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.float16 if args.half else torch.float32
    )

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    config = peft.LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=args.lora_target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=peft.TaskType.CAUSAL_LM,
    )

    model = peft.get_peft_model(model, config)


    train_dataset = load_dataset(path=args.data_path, split="train").shuffle()
    test_dataset = load_dataset(path=args.data_path, split="test").shuffle()
    train_data = ConstantLengthDataset(
      tokenizer,
      train_dataset,
      input_column_name="src_fm",
      output_column_name="target"
    )
    test_data = ConstantLengthDataset(
      tokenizer,
      test_dataset,
      input_column_name="src_fm",
      output_column_name="target"
    )


    resume_from_checkpoint = args.resume_from_checkpoint
    if resume_from_checkpoint:
        # Check the available weights and load them
        checkpoint_name = os.path.join(
            resume_from_checkpoint, "pytorch_model.bin"
        )  # Full checkpoint
        if not os.path.exists(checkpoint_name):
            checkpoint_name = os.path.join(
                resume_from_checkpoint, "adapter_model.bin"
            )  # only LoRA model - LoRA config above has to fit
            resume_from_checkpoint = False  # So the trainer won't try loading its state
        # The two files above have a different name depending on how they were saved, but are actually the same.
        if os.path.exists(checkpoint_name):
            print(f"Restarting from {checkpoint_name}")
            adapters_weights = torch.load(checkpoint_name)
            model = peft.set_peft_model_state_dict(model, adapters_weights)
        else:
            print(f"Checkpoint {checkpoint_name} not found")

    model.print_trainable_parameters()  # Be more transparent about the % of trainable params.

    # train_val = data.train_test_split(
    #     test_size=args.val_set_size, shuffle=True, seed=42
    # )
    # train_data = train_val["train"].shuffle()
    # val_data = train_val["test"].shuffle()

    # Check if parameter passed or if set within environ
    use_wandb = len(args.wandb_project) > 0 or (
        "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
    )
    # Only overwrite environ if wandb param passed
    print("args.wandb_log_model=",args.wandb_log_model)
    if len(args.wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = args.wandb_project
    if len(args.wandb_watch) > 0:
        os.environ["WANDB_WATCH"] = args.wandb_watch
    if len(args.wandb_log_model) > 0:
        os.environ["WANDB_LOG_MODEL"] = args.wandb_log_model

    training_agrs = TrainingArguments(
            per_device_train_batch_size=args.micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=100,
            num_train_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            fp16=args.half,
            logging_steps=10,
            evaluation_strategy="steps",
            save_strategy="steps",
            max_steps=args.max_steps,
            eval_steps=args.eval_steps,
            save_steps=args.eval_steps,
            output_dir=args.output_dir,
            save_total_limit=3,
            report_to="wandb" if use_wandb else None,
            run_name=args.wandb_run_name,
            load_best_model_at_end=True,
        )

    trainer = Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=test_data,
        args=training_agrs,
    )
    model.config.use_cache = False

    old_state_dict = model.state_dict
    model.state_dict = (
        lambda self, *_, **__: peft.get_peft_model_state_dict(self, old_state_dict())
    ).__get__(model, type(model))

    model = torch.compile(model)

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    model.save_pretrained(args.output_dir)

    print("\n If there's a warning about missing keys above, please disregard :)")

if __name__ == "__main__":
    args = parse_args()
    print(args)
    train(args)