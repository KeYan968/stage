"""
Fine-tuned model inference — LLaMA 3.2-3B-Instruct after QLoRA fine-tuning.
Output in QALD format for GERBIL evaluation.

Usage:
    python3 main_sparql_test-v2.py
"""

import re, torch, json, time, os
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from SPARQLWrapper import SPARQLWrapper, JSON, SPARQLExceptions

# ── Paths ──────────────────────────────────────────────────────────────────────
MODEL_PATH  = "/infmodels/kyan/Llama-SPARQL-Correction-V2-ErrorType/merged_model"
TEST_FILE   = "/home/kyan/stage/Datasets/Generated_wrong_sparql_datasets/test_set_updated.json"
OUTPUT_FILE = "/home/kyan/stage/Datasets/v2_results/finetuned_v2_errortype_gerbil.json"
ENDPOINT    = "https://dbpedia.org/sparql"
TIMEOUT     = 15

# ── SPARQL execution → QALD-format answers block ──────────────────────────────
def execute_query(query):
    sparql = SPARQLWrapper(ENDPOINT)
    sparql.setTimeout(TIMEOUT)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    try:
        qres     = sparql.query().convert()
        head     = qres.get("head", {"link": [], "vars": []})
        bindings = qres.get("results", {}).get("bindings", [])
        return [{
            "head"   : {"link": head.get("link", []), "vars": head.get("vars", [])},
            "results": {"distinct": False, "ordered": True, "bindings": bindings},
        }]
    except Exception:
        return [{
            "head"   : {"link": [], "vars": []},
            "results": {"distinct": False, "ordered": True, "bindings": []},
        }]

# ── Extract clean SPARQL from model output ─────────────────────────────────────
def extract_sparql(text):
    if "assistant" in text:
        text = text.split("assistant")[-1].strip()
    # Try ```sparql ... ``` code block first
    match = re.search(r'```(?:sparql)?\s*(.*?)```', text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    # Fallback: first line starting with a SPARQL keyword
    for i, line in enumerate(text.strip().splitlines()):
        if any(line.strip().upper().startswith(kw) for kw in ("SELECT", "ASK", "CONSTRUCT", "DESCRIBE", "PREFIX")):
            return "\n".join(text.strip().splitlines()[i:]).strip()
    return text.strip()

# ── Load model & tokenizer ─────────────────────────────────────────────────────
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=bnb_config,
    device_map="auto",
    attn_implementation="eager",
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id
model.config.use_cache = True

terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>"),
]

# ── Load test set ──────────────────────────────────────────────────────────────
with open(TEST_FILE, "r", encoding="utf-8") as f:
    test_data = json.load(f)

# ── Inference + SPARQL execution loop ─────────────────────────────────────────
questions_output = []

for i, item in enumerate(test_data):
    print(f"[{i+1}/{len(test_data)}] {item['question'][:60]}...")

    messages = [{
    "role": "user",
    "content": (
        f"Question: {item['question']}\n"
        f"Wrong SPARQL query: {item['sparql_wrong']}\n"
        f"Error type: {', '.join(item['error_types']) if item.get('error_types') else 'unknown'}\n"
        "Output the corrected SPARQL query only, with no explanation or commentary."
    ),
    }]

    prompt  = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs  = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to("cuda")
    outputs = model.generate(
        **inputs,
        num_return_sequences=1,
        eos_token_id=terminators,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )

    text            = tokenizer.decode(outputs[0], skip_special_tokens=True)
    predicted_query = extract_sparql(text)
    print(f"  Query : {predicted_query[:80]}...")

    time.sleep(0.3)
    answers = execute_query(predicted_query)
    print(f"  Result: {len(answers[0]['results']['bindings'])} bindings")

    qid = str(item.get("id", i)).lstrip("q")
    questions_output.append({
        "id"      : qid,
        "question": [{"language": "en", "string": item["question"]}],
        "query"   : {"sparql": predicted_query},
        "answers" : answers,
    })

# ── Save QALD-format output ────────────────────────────────────────────────────
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump({"questions": questions_output}, f, indent=4, ensure_ascii=False)

print(f"\nDone. Saved to '{OUTPUT_FILE}'.")