"""
Fine-tuned model inference — LLaMA 3-8B-Instruct after QLoRA fine-tuning (V2, with error_type).
Runs the SAME inference 5 times with identical parameters (for reliability analysis).
Model is loaded once; only generation is repeated.
Output in QALD format for GERBIL evaluation.

Usage:
    python3 main_sparql_test_v2_5runs.py
"""

import re, torch, json, time, os
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from SPARQLWrapper import SPARQLWrapper, JSON, SPARQLExceptions

# ── Paths ──────────────────────────────────────────────────────────────────────
MODEL_PATH  = "/home/kyan/stage/scripts/Fintuning/Llama-SPARQL-Correction_Llama-3-8B-Instruct_v2/merged_model"
TEST_FILE   = "/home/kyan/stage/Datasets/Generated_wrong_sparql_datasets/test_set_updated.json"
OUTPUT_DIR  = "/home/kyan/stage/Datasets/Experiment_results/LLaMA 3-8B-Instruct/v2_results_LLaMA 3-8B-Instruct/5runs"
N_RUNS      = 5
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
    match = re.search(r'```(?:sparql)?\s*(.*?)```', text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    for i, line in enumerate(text.strip().splitlines()):
        if any(line.strip().upper().startswith(kw) for kw in ("SELECT", "ASK", "CONSTRUCT", "DESCRIBE", "PREFIX")):
            return "\n".join(text.strip().splitlines()[i:]).strip()
    return text.strip()

# ── Load model & tokenizer (once) ──────────────────────────────────────────────
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

# ── Load test set (once) ───────────────────────────────────────────────────────
with open(TEST_FILE, "r", encoding="utf-8") as f:
    test_data = json.load(f)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Run inference N_RUNS times with identical parameters ──────────────────────
for run in range(1, N_RUNS + 1):
    print(f"\n{'='*60}")
    print(f"RUN {run}/{N_RUNS}")
    print(f"{'='*60}")

    questions_output = []

    for i, item in enumerate(test_data):
        print(f"[Run {run}][{i+1}/{len(test_data)}] {item['question'][:50]}...")

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

        time.sleep(0.3)
        answers = execute_query(predicted_query)

        qid = str(item.get("id", i)).lstrip("q")
        questions_output.append({
            "id"      : qid,
            "question": [{"language": "en", "string": item["question"]}],
            "query"   : {"sparql": predicted_query},
            "answers" : answers,
        })

    output_file = os.path.join(OUTPUT_DIR, f"v2_run_{run}.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({"questions": questions_output}, f, indent=4, ensure_ascii=False)
    print(f"\nRun {run} saved to '{output_file}'.")

print(f"\nAll {N_RUNS} runs complete. Results saved in '{OUTPUT_DIR}/'.")
