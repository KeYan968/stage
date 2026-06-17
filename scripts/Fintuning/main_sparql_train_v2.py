"""
Fine-tuning V2 — adds error_types to the prompt.

Task  : NL question + wrong SPARQL + error type  →  corrected SPARQL
Fields: question, sparql_wrong, sparql_gold, error_types

Usage:
    python3 main_sparql_train_v2.py
"""

import os, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, PeftModel
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_MODEL  = "/home/kyan/models/Llama-3.2-3B-Instruct"
TRAIN_FILE  = "/home/kyan/stage/Datasets/Generated_wrong_sparql_datasets/train_set_updated.json"

MODEL_NAME   = "Llama-SPARQL-Correction-V2-ErrorType"
RESULTS_DIR  = "/infmodels/kyan/" + MODEL_NAME + "/"
ADAPTER_PATH = RESULTS_DIR + "adapter"
OUTPUT_PATH  = RESULTS_DIR + "merged_model"

# ── Hyperparameters ────────────────────────────────────────────────────────────
EPOCHS      = 6
LR          = 2e-4
BATCH_SIZE  = 1
GRAD_ACCUM  = 2
MAX_SEQ_LEN = 512
LORA_R      = 16
LORA_ALPHA  = 32

# ── Model & tokenizer ──────────────────────────────────────────────────────────
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
    attn_implementation="eager",
)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id

# ── LoRA ───────────────────────────────────────────────────────────────────────
peft_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["up_proj", "down_proj", "gate_proj", "k_proj", "q_proj", "v_proj", "o_proj"],
)

# ── Dataset ────────────────────────────────────────────────────────────────────
dataset = load_dataset("json", data_files={"train": TRAIN_FILE})

def format_chat_template(row):
    # V2: add error_types to prompt
    error_type_str = ", ".join(row["error_types"]) if row.get("error_types") else "unknown"
    row_json = [
        {
            "role": "user",
            "content": (
                f"Question: {row['question']}\n"
                f"Wrong SPARQL query: {row['sparql_wrong']}\n"
                f"Error type: {error_type_str}\n"
                "Output the corrected SPARQL query only, with no explanation or commentary."
            ),
        },
        {
            "role": "assistant",
            "content": row["sparql_gold"] + "<|eot_id|>",
        },
    ]
    row["text"] = tokenizer.apply_chat_template(row_json, tokenize=False)
    return row

dataset = dataset.map(format_chat_template, num_proc=4)

# ── Training ───────────────────────────────────────────────────────────────────
sft_config = SFTConfig(
    output_dir=OUTPUT_PATH,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    optim="paged_adamw_32bit",
    num_train_epochs=EPOCHS,
    eval_strategy="no",
    logging_steps=1,
    warmup_steps=10,
    logging_strategy="steps",
    learning_rate=LR,
    fp16=False,
    bf16=False,
    group_by_length=True,
    max_grad_norm=0.3,
    max_length=MAX_SEQ_LEN,
    dataset_text_field="text",
    packing=False,
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    peft_config=peft_config,
    processing_class=tokenizer,
    args=sft_config,
)

trainer.train()
model.config.use_cache = True

# ── Save adapter ───────────────────────────────────────────────────────────────
trainer.model.save_pretrained(ADAPTER_PATH)

# ── Merge adapter into base model ──────────────────────────────────────────────
tokenizer_reload = AutoTokenizer.from_pretrained(BASE_MODEL)
if tokenizer_reload.pad_token is None:
    tokenizer_reload.pad_token = tokenizer_reload.eos_token

base_model_reload = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    return_dict=True,
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)

final_model = PeftModel.from_pretrained(base_model_reload, ADAPTER_PATH)
final_model = final_model.merge_and_unload()

final_model.save_pretrained(OUTPUT_PATH)
tokenizer_reload.save_pretrained(OUTPUT_PATH)

# Uncomment to push to Hub:
# from huggingface_hub import login
# from dotenv import load_dotenv
# load_dotenv()
# login(token=os.getenv("HUGGINGFACE_TOKEN"))
# final_model.push_to_hub(MODEL_NAME, use_temp_dir=False)
# tokenizer_reload.push_to_hub(MODEL_NAME, use_temp_dir=False)

print(f"Done. Model saved to '{OUTPUT_PATH}'.")

