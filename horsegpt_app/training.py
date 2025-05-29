from unsloth import FastLanguageModel
from datasets import Dataset
import torch
from trl import SFTTrainer
from transformers import TrainingArguments
import json

# 1. Load base model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "meta-llama/Llama-2-7b-chat-hf", # or use your existing model
    max_seq_length = 2048,
    dtype = None,
    load_in_4bit = True,
)

# 2. Add LoRA adapters 
"""Instead of updating ALL 7 billion parameters (which requires massive compute), 
LoRA adds small "adapter" layers that learn the differences for your specific task.""" 

model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Higher = more parameters, better quality but slower
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
)

# 3. Load your horse racing dataset
def load_dataset(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return Dataset.from_list(data)

dataset = load_dataset('horse_racing_data.jsonl')

# 4. Format prompts
def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    for instruction, input_text, output in zip(instructions, inputs, outputs):
        text = f"<s>[INST] {instruction}"
        if input_text:
            text += f"\n{input_text}"
        text += f" [/INST] {output}</s>"
        texts.append(text)
    return { "text" : texts, }

dataset = dataset.map(formatting_prompts_func, batched = True,)

# 5. Train the model
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = 2048,
    dataset_num_proc = 2,
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 100, # Increase for better results
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "horse_racing_model",
        save_steps = 50,
    ),
)

# Start training
trainer.train()

# Save the model
model.save_pretrained("horse_racing_lora")
tokenizer.save_pretrained("horse_racing_lora")