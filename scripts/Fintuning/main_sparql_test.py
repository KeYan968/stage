"""
Batch inference for SPARQL correction on the test set.
Inspired by the supervisor's translate.py, adapted for batch processing.

For each test item:
  - Generates a corrected SPARQL query
  - Validates it against DBpedia SPARQL endpoint
  - Retries up to WRONG_QUERY_MAX times if syntax error

Usage:
    python3 main_sparql_test.py
"""

import torch, json
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from SPARQLWrapper import SPARQLWrapper, JSON, SPARQLExceptions

# ── Paths ──────────────────────────────────────────────────────────────────────
MODEL_PATH  = "Llama-SPARQL-Correction/merged_model"
TEST_FILE   = "/home/kyan/stage/Datasets/Generated_wrong_sparql_datasets/test_set_updated.json"
OUTPUT_FILE = "Llama-SPARQL-Correction/test_predictions.json"

# ── Config ─────────────────────────────────────────────────────────────────────
ENDPOINT       = "https://dbpedia.org/sparql"
WRONG_QUERY_MAX = 10

# ── SPARQL validation ──────────────────────────────────────────────────────────
def querying(query):
    sparql = SPARQLWrapper(ENDPOINT)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    try:
        qres = sparql.query().convert()
        if not qres.get("results", {}).get("bindings"):
            return ("no_results", None)
        return ("success", qres)
    except SPARQLExceptions.QueryBadFormed as e:
        return ("syntax_error", str(e))
    except Exception as e:
        return ("error", str(e))

# ── Load model ─────────────────────────────────────────────────────────────────
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

# ── Inference loop ─────────────────────────────────────────────────────────────
results = []

for i, item in enumerate(test_data):
    print(f"\n[{i+1}/{len(test_data)}] Question: {item['question'][:60]}...")

    messages = [
        {
            "role": "user",
            "content": (
                f"Question: {item['question']}\n"
                f"Wrong SPARQL query: {item['sparql_wrong']}\n"
                "Generate the corrected SPARQL query."
            ),
        }
    ]

    wrong_queries_count = 0
    translation_ended = False
    final_query = None
    final_status = None

    while not translation_ended and wrong_queries_count < WRONG_QUERY_MAX:
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to("cuda")

        outputs = model.generate(
            **inputs,
            num_return_sequences=1,
            eos_token_id=terminators,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )

        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_query = text.split("assistant")[-1].strip()
        print(f"  Attempt {wrong_queries_count+1}: {generated_query[:80]}...")

        status, _ = querying(generated_query)
        final_query = generated_query
        final_status = status

        if status not in {"syntax_error", "error"}:
            translation_ended = True
        else:
            wrong_queries_count += 1
            if wrong_queries_count < WRONG_QUERY_MAX:
                print(f"  Status: {status} — retrying...")

    print(f"  Final status: {final_status} (attempts: {wrong_queries_count+1})")

    results.append({
        "id": item.get("id", i),
        "question": item["question"],
        "sparql_wrong": item["sparql_wrong"],
        "sparql_gold": item.get("sparql_gold", ""),
        "sparql_predicted": final_query,
        "sparql_status": final_status,
        "attempts": wrong_queries_count + 1,
    })

# ── Save results ───────────────────────────────────────────────────────────────
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4, ensure_ascii=False)

print(f"\nDone. Results saved to '{OUTPUT_FILE}'.")
