"""
Evaluation script for SPARQL correction results in QALD format.

Reads two QALD files: predictions and gold, computes all metrics.

Metrics:
  GERBIL-style:
    Micro F1 / Precision / Recall
    Macro F1 / Precision / Recall
    Macro F1-QALD
  Additional:
    Exact Match (EM)
    Execution Success Rate
    Answer Match (AM)
    Answer F1-QALD (per-item)
    BLEU-1
    Property EM (Pid-EM)

Usage:
    python3 evaluate_qald.py
"""

import json, re, time
from collections import Counter
from SPARQLWrapper import SPARQLWrapper, JSON, SPARQLExceptions

# ── Paths — edit before each run ──────────────────────────────────────────────
#PRED_FILE   = "results/finetuned_gerbil.json"   # your predicted QALD file
GOLD_FILE   = "/home/kyan/stage/Datasets/Generated_wrong_sparql_datasets/test_set_updated.json"
#OUTPUT_FILE = "results/finetuned_evaluation.json"

# For baseline:
#PRED_FILE   = "results/baseline_gerbil.json"
#OUTPUT_FILE = "results/baseline_evaluation.json"

# For v2 with error types model
# PRED_FILE   = "/home/kyan/stage/Datasets/v2_results/finetuned_v2_errortype_gerbil.json"
# OUTPUT_FILE = "/home/kyan/stage/Datasets/v2_results/finetuned_v2_errortype_evaluation.json"

# For v2 with fewshot
#PRED_FILE   = "/home/kyan/stage/Datasets/Experiment_results/v2_fewshot_results/finetuned_v2_fewshot_gerbil.json"
#OUTPUT_FILE = "/home/kyan/stage/Datasets/Experiment_results/v2_fewshot_results/finetuned_v2_fewshot_evaluation.json"

# For v1 with fewshot
#PRED_FILE   = "/home/kyan/stage/Datasets/Experiment_results/v1_fewshot_results/finetuned_v1_fewshot_gerbil.json"
#OUTPUT_FILE = "/home/kyan/stage/Datasets/Experiment_results/v1_fewshot_results/finetuned_v1_fewshot_evaluation.json"

# For LLama 3 8b Model
# For baseline:
#PRED_FILE   = "/home/kyan/stage/Datasets/Experiment_results/LLaMA 3-8B-Instruct/Baseline_results_LLaMA 3-8B-Instruct/baseline_gerbil.json"
#OUTPUT_FILE = "/home/kyan/stage/Datasets/Experiment_results/LLaMA 3-8B-Instruct/Baseline_results_LLaMA 3-8B-Instruct/baseline_evaluation.json"

# For v1 without errortype
#PRED_FILE   = "/home/kyan/stage/Datasets/Experiment_results/LLaMA 3-8B-Instruct/v1_results_LLaMA 3-8B-Instruct/v1_results_LLaMA 3-8B-Instruct_gerbil.json"
#OUTPUT_FILE = "/home/kyan/stage/Datasets/Experiment_results/LLaMA 3-8B-Instruct/v1_results_LLaMA 3-8B-Instruct/v1_evaluation.json"

# For v2 with error types model
#PRED_FILE   = "/home/kyan/stage/Datasets/Experiment_results/LLaMA 3-8B-Instruct/v2_results_LLaMA 3-8B-Instruct/v2_results_LLaMA 3-8B-Instruct_gerbil.json"
#OUTPUT_FILE = "/home/kyan/stage/Datasets/Experiment_results/LLaMA 3-8B-Instruct/v2_results_LLaMA 3-8B-Instruct/v2_errortype_evaluation.json"

# For v2 with fewshot
PRED_FILE   = "/home/kyan/stage/Datasets/Experiment_results/LLaMA 3-8B-Instruct/v2_fewshot_results_LLaMA 3-8B-Instruct/v2_fewshot_results_LLaMA 3-8B-Instruct_gerbil.json"
OUTPUT_FILE = "/home/kyan/stage/Datasets/Experiment_results/LLaMA 3-8B-Instruct/v2_fewshot_results_LLaMA 3-8B-Instruct/v2_fewshot_evaluation.json"

# For v1 with fewshot
#PRED_FILE   = "/home/kyan/stage/Datasets/Experiment_results/LLaMA 3-8B-Instruct/v1_fewshot_results_LLaMA 3-8B-Instruct/v1_fewshot_results_LLaMA 3-8B-Instruct_gerbil.json"
#OUTPUT_FILE = "/home/kyan/stage/Datasets/Experiment_results/LLaMA 3-8B-Instruct/v1_fewshot_results_LLaMA 3-8B-Instruct/v1_fewshot_evaluation.json"

ENDPOINT = "https://dbpedia.org/sparql"
TIMEOUT  = 15

# ── SPARQL execution ───────────────────────────────────────────────────────────
def execute_query(query):
    """Execute query, return (status, answer_set of URI/literal strings)."""
    sparql = SPARQLWrapper(ENDPOINT)
    sparql.setTimeout(TIMEOUT)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    try:
        qres     = sparql.query().convert()
        bindings = qres.get("results", {}).get("bindings", [])
        answer_set = set()
        for binding in bindings:
            for val in binding.values():
                answer_set.add(val.get("value", "").strip())
        return ("success", answer_set)
    except SPARQLExceptions.QueryBadFormed:
        return ("syntax_error", set())
    except Exception:
        return ("error", set())

# ── Extract answer set from QALD bindings ─────────────────────────────────────
def bindings_to_set(answers):
    """Flatten QALD answers block into a set of value strings."""
    result = set()
    for ans_block in answers:
        for binding in ans_block.get("results", {}).get("bindings", []):
            for val in binding.values():
                result.add(val.get("value", "").strip())
    return result

# ── String metrics ─────────────────────────────────────────────────────────────
def normalize_query(q):
    return re.sub(r'\s+', ' ', q.strip().lower())

def tokenize(text):
    return re.findall(r'\S+', text.lower())

def bleu1(pred, gold):
    pred_tok = tokenize(pred)
    gold_tok = tokenize(gold)
    if not pred_tok:
        return 0.0
    gold_c = Counter(gold_tok)
    matches = sum(min(c, gold_c[t]) for t, c in Counter(pred_tok).items())
    return matches / len(pred_tok)

def extract_properties(query):
    return set(re.findall(r'<http://dbpedia\.org/(?:ontology|property)/[^>]+>', query))

# ── Per-item F1 between two answer sets ───────────────────────────────────────
def set_f1(pred_set, gold_set):
    if not gold_set and not pred_set:
        return 1.0, 1.0, 1.0
    if not pred_set:
        return 0.0, 0.0, 0.0
    if not gold_set:
        return 0.0, 0.0, 0.0
    tp        = len(pred_set & gold_set)
    precision = tp / len(pred_set)
    recall    = tp / len(gold_set)
    f1        = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return f1, precision, recall

# ── Load files ─────────────────────────────────────────────────────────────────
with open(PRED_FILE, "r", encoding="utf-8") as f:
    pred_data = json.load(f)

with open(GOLD_FILE, "r", encoding="utf-8") as f:
    gold_raw = json.load(f)

# Build gold lookup by id (strip "q" prefix to match QALD format)
gold_lookup = {}
for item in gold_raw:
    qid = str(item["id"]).lstrip("q")
    gold_lookup[qid] = item

pred_questions = pred_data["questions"]
total = len(pred_questions)
print(f"Evaluating {total} predictions...\n")

# ── Per-item results ───────────────────────────────────────────────────────────
per_item = []

# Counters for GERBIL-style micro metrics
micro_tp = micro_fp = micro_fn = 0

# Accumulators for macro metrics
macro_f1_sum = macro_p_sum = macro_r_sum = 0.0
macro_f1_qald_sum = 0.0

# Accumulators for additional metrics
em_count = exec_success = answer_match = 0
total_bleu = total_ans_f1 = pid_em_count = 0.0

for i, pred_item in enumerate(pred_questions):
    qid        = str(pred_item["id"])
    pred_query = pred_item["query"]["sparql"].strip()
    pred_answers_set = bindings_to_set(pred_item.get("answers", []))

    gold_item  = gold_lookup.get(qid, {})
    gold_query = gold_item.get("sparql_gold", "").strip()

    # ── Get gold answer set by executing gold query ────────────────────────────
    time.sleep(0.2)
    gold_status, gold_answers_set = execute_query(gold_query)

    # ── Execution success of predicted query ───────────────────────────────────
    time.sleep(0.2)
    pred_exec_status, pred_exec_set = execute_query(pred_query)
    if pred_exec_status == "success":
        exec_success += 1

    # ── GERBIL-style Micro (aggregate TP/FP/FN) ───────────────────────────────
    tp = len(pred_answers_set & gold_answers_set)
    fp = len(pred_answers_set - gold_answers_set)
    fn = len(gold_answers_set - pred_answers_set)
    micro_tp += tp
    micro_fp += fp
    micro_fn += fn

    # ── Macro per-item F1/P/R ─────────────────────────────────────────────────
    item_f1, item_p, item_r = set_f1(pred_answers_set, gold_answers_set)
    macro_f1_sum += item_f1
    macro_p_sum  += item_p
    macro_r_sum  += item_r

    # ── Macro F1-QALD (empty pred & empty gold = 1, else standard F1) ─────────
    macro_f1_qald_sum += item_f1   # same as macro F1 in QALD convention

    # ── Exact Match ───────────────────────────────────────────────────────────
    em = int(normalize_query(pred_query) == normalize_query(gold_query))
    em_count += em

    # ── BLEU-1 ────────────────────────────────────────────────────────────────
    bleu = bleu1(pred_query, gold_query)
    total_bleu += bleu

    # ── Property EM ───────────────────────────────────────────────────────────
    pid_em = int(extract_properties(pred_query) == extract_properties(gold_query))
    pid_em_count += pid_em

    # ── Answer Match ──────────────────────────────────────────────────────────
    am = int(pred_answers_set == gold_answers_set
             and pred_exec_status not in ("error",)
             and gold_status not in ("error",))
    answer_match += am

    # ── Answer F1-QALD (uses executed answers, not stored answers) ────────────
    ans_f1, _, _ = set_f1(pred_exec_set, gold_answers_set)
    total_ans_f1 += ans_f1

    per_item.append({
        "id"            : qid,
        "exact_match"   : em,
        "bleu1"         : round(bleu, 4),
        "property_em"   : pid_em,
        "exec_status"   : pred_exec_status,
        "answer_match"  : am,
        "answer_f1"     : round(ans_f1, 4),
        "macro_f1"      : round(item_f1, 4),
        "macro_precision": round(item_p, 4),
        "macro_recall"  : round(item_r, 4),
    })

    print(f"[{i+1}/{total}] id={qid} | EM={em} | macro_F1={item_f1:.2f} | exec={pred_exec_status}")

# ── Micro metrics ──────────────────────────────────────────────────────────────
micro_p = micro_tp / (micro_tp + micro_fp) if (micro_tp + micro_fp) > 0 else 0.0
micro_r = micro_tp / (micro_tp + micro_fn) if (micro_tp + micro_fn) > 0 else 0.0
micro_f1 = (2 * micro_p * micro_r / (micro_p + micro_r)) if (micro_p + micro_r) > 0 else 0.0

# ── Summary ────────────────────────────────────────────────────────────────────
summary = {
    "total"                  : total,
    # GERBIL-style
    "micro_f1"               : round(micro_f1, 4),
    "micro_precision"        : round(micro_p, 4),
    "micro_recall"           : round(micro_r, 4),
    "macro_f1"               : round(macro_f1_sum / total, 4),
    "macro_precision"        : round(macro_p_sum / total, 4),
    "macro_recall"           : round(macro_r_sum / total, 4),
    "macro_f1_qald"          : round(macro_f1_qald_sum / total, 4),
    # Additional
    "exact_match"            : round(em_count / total, 4),
    "execution_success_rate" : round(exec_success / total, 4),
    "answer_match"           : round(answer_match / total, 4),
    "answer_f1_qald"         : round(total_ans_f1 / total, 4),
    "bleu1"                  : round(total_bleu / total, 4),
    "property_em"            : round(pid_em_count / total, 4),
}

print("\n" + "="*55)
print("EVALUATION SUMMARY")
print("="*55)
for k, v in summary.items():
    print(f"  {k:<30} {v}")
print("="*55)

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump({"summary": summary, "per_item": per_item}, f, indent=4, ensure_ascii=False)

print(f"\nFull results saved to '{OUTPUT_FILE}'.")