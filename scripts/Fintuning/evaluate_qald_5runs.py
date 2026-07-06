"""
Evaluation script for 5-run reliability analysis.
Evaluates 5 prediction files from the same configuration, computes per-run
metrics plus mean ± std across runs.

Edit CONFIG below to switch between: baseline, v1, v1_fewshot, v2, v2_fewshot.

Usage:
    python3 evaluate_qald_5runs.py
"""

import json, re, time, os
from collections import Counter
from SPARQLWrapper import SPARQLWrapper, JSON, SPARQLExceptions

# ── Config — edit this block before each run ───────────────────────────────────
CONFIG = "v2_fewshot"   # one of: "baseline", "v1", "v1_fewshot", "v2", "v2_fewshot"

CONFIGS = {
    "baseline": {
        "pred_dir"   : "/home/kyan/stage/Datasets/Experiment_results/LLaMA 3-8B-Instruct/Baseline_results_LLaMA 3-8B-Instruct/5runs",
        "pred_prefix": "baseline_run",
        "output_file": "/home/kyan/stage/Datasets/Experiment_results/LLaMA 3-8B-Instruct/baseline_5runs_summary.json",
    },
    "v1": {
        "pred_dir"   : "/home/kyan/stage/Datasets/Experiment_results/LLaMA 3-8B-Instruct/v1_results_LLaMA 3-8B-Instruct/5runs",
        "pred_prefix": "v1_run",
        "output_file": "/home/kyan/stage/Datasets/Experiment_results/LLaMA 3-8B-Instruct/v1_5runs_summary.json",
    },
    "v1_fewshot": {
        "pred_dir"   : "/home/kyan/stage/Datasets/Experiment_results/LLaMA 3-8B-Instruct/v1_fewshot_results_LLaMA 3-8B-Instruct/5runs",
        "pred_prefix": "v1_fewshot_run",
        "output_file": "/home/kyan/stage/Datasets/Experiment_results/LLaMA 3-8B-Instruct/v1_fewshot_5runs_summary.json",
    },
    "v2": {
        "pred_dir"   : "/home/kyan/stage/Datasets/Experiment_results/LLaMA 3-8B-Instruct/v2_results_LLaMA 3-8B-Instruct/5runs",
        "pred_prefix": "v2_run",
        "output_file": "/home/kyan/stage/Datasets/Experiment_results/LLaMA 3-8B-Instruct/v2_5runs_summary.json",
    },
    "v2_fewshot": {
        "pred_dir"   : "/home/kyan/stage/Datasets/Experiment_results/LLaMA 3-8B-Instruct/v2_fewshot_results_LLaMA 3-8B-Instruct/5runs",
        "pred_prefix": "v2_fewshot_run",
        "output_file": "/home/kyan/stage/Datasets/Experiment_results/LLaMA 3-8B-Instruct/v2_fewshot_5runs_summary.json",
    },
}

PRED_DIR    = CONFIGS[CONFIG]["pred_dir"]
PRED_PREFIX = CONFIGS[CONFIG]["pred_prefix"]
OUTPUT_FILE = CONFIGS[CONFIG]["output_file"]

GOLD_FILE = "/home/kyan/stage/Datasets/Generated_wrong_sparql_datasets/test_set_updated.json"
N_RUNS    = 5

ENDPOINT = "https://dbpedia.org/sparql"
TIMEOUT  = 15

# ── SPARQL execution ───────────────────────────────────────────────────────────
def execute_query(query):
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

def bindings_to_set(answers):
    result = set()
    for ans_block in answers:
        for binding in ans_block.get("results", {}).get("bindings", []):
            for val in binding.values():
                result.add(val.get("value", "").strip())
    return result

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

def set_f1(pred_set, gold_set):
    if not gold_set and not pred_set:
        return 1.0, 1.0, 1.0
    if not pred_set or not gold_set:
        return 0.0, 0.0, 0.0
    tp        = len(pred_set & gold_set)
    precision = tp / len(pred_set)
    recall    = tp / len(gold_set)
    f1        = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return f1, precision, recall

# ── Load gold once ──────────────────────────────────────────────────────────────
with open(GOLD_FILE, "r", encoding="utf-8") as f:
    gold_raw = json.load(f)
gold_lookup = {str(item["id"]).lstrip("q"): item for item in gold_raw}

# Cache gold query executions across runs (gold doesn't change between runs)
gold_answer_cache = {}

def get_gold_answers(qid, gold_query):
    if qid not in gold_answer_cache:
        time.sleep(0.2)
        status, ans_set = execute_query(gold_query)
        gold_answer_cache[qid] = (status, ans_set)
    return gold_answer_cache[qid]

# ── Evaluate one run ────────────────────────────────────────────────────────────
def evaluate_run(pred_file):
    with open(pred_file, "r", encoding="utf-8") as f:
        pred_data = json.load(f)
    pred_questions = pred_data["questions"]
    total = len(pred_questions)

    micro_tp = micro_fp = micro_fn = 0
    macro_f1_sum = macro_p_sum = macro_r_sum = 0.0
    em_count = exec_success = answer_match = 0
    total_bleu = total_ans_f1 = pid_em_count = 0.0

    for pred_item in pred_questions:
        qid        = str(pred_item["id"])
        pred_query = pred_item["query"]["sparql"].strip()
        pred_answers_set = bindings_to_set(pred_item.get("answers", []))

        gold_item  = gold_lookup.get(qid, {})
        gold_query = gold_item.get("sparql_gold", "").strip()

        gold_status, gold_answers_set = get_gold_answers(qid, gold_query)

        time.sleep(0.2)
        pred_exec_status, pred_exec_set = execute_query(pred_query)
        if pred_exec_status == "success":
            exec_success += 1

        tp = len(pred_answers_set & gold_answers_set)
        fp = len(pred_answers_set - gold_answers_set)
        fn = len(gold_answers_set - pred_answers_set)
        micro_tp += tp; micro_fp += fp; micro_fn += fn

        item_f1, item_p, item_r = set_f1(pred_answers_set, gold_answers_set)
        macro_f1_sum += item_f1
        macro_p_sum  += item_p
        macro_r_sum  += item_r

        em = int(normalize_query(pred_query) == normalize_query(gold_query))
        em_count += em

        total_bleu += bleu1(pred_query, gold_query)

        pid_em = int(extract_properties(pred_query) == extract_properties(gold_query))
        pid_em_count += pid_em

        am = int(pred_answers_set == gold_answers_set
                 and pred_exec_status != "error" and gold_status != "error")
        answer_match += am

        ans_f1, _, _ = set_f1(pred_exec_set, gold_answers_set)
        total_ans_f1 += ans_f1

    micro_p = micro_tp / (micro_tp + micro_fp) if (micro_tp + micro_fp) > 0 else 0.0
    micro_r = micro_tp / (micro_tp + micro_fn) if (micro_tp + micro_fn) > 0 else 0.0
    micro_f1 = (2 * micro_p * micro_r / (micro_p + micro_r)) if (micro_p + micro_r) > 0 else 0.0

    return {
        "total"                  : total,
        "micro_f1"               : round(micro_f1, 4),
        "micro_precision"        : round(micro_p, 4),
        "micro_recall"           : round(micro_r, 4),
        "macro_f1"               : round(macro_f1_sum / total, 4),
        "macro_precision"        : round(macro_p_sum / total, 4),
        "macro_recall"           : round(macro_r_sum / total, 4),
        "macro_f1_qald"          : round(macro_f1_sum / total, 4),
        "exact_match"            : round(em_count / total, 4),
        "execution_success_rate" : round(exec_success / total, 4),
        "answer_match"           : round(answer_match / total, 4),
        "answer_f1_qald"         : round(total_ans_f1 / total, 4),
        "bleu1"                  : round(total_bleu / total, 4),
        "property_em"            : round(pid_em_count / total, 4),
    }

# ── Evaluate all 5 runs ──────────────────────────────────────────────────────────
print(f"Evaluating configuration: {CONFIG}")
print(f"Prediction directory: {PRED_DIR}\n")

all_runs = {}
for run in range(1, N_RUNS + 1):
    pred_file = os.path.join(PRED_DIR, f"{PRED_PREFIX}_{run}.json")
    print(f"\nEvaluating run {run}: {pred_file}")
    summary = evaluate_run(pred_file)
    all_runs[f"run_{run}"] = summary
    print(f"  Run {run} summary: {summary}")

# ── Compute mean ± std across runs ────────────────────────────────────────────
metrics = list(all_runs["run_1"].keys())
metrics.remove("total")

mean_std = {}
for m in metrics:
    values = [all_runs[f"run_{r}"][m] for r in range(1, N_RUNS + 1)]
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    std = variance ** 0.5
    mean_std[m] = {"mean": round(mean, 4), "std": round(std, 4), "values": values}

# ── Print summary table ────────────────────────────────────────────────────────
print("\n" + "="*70)
print(f"5-RUN RELIABILITY SUMMARY — {CONFIG}")
print("="*70)
print(f"{'Metric':<25} {'Run1':<8}{'Run2':<8}{'Run3':<8}{'Run4':<8}{'Run5':<8} {'Mean±Std'}")
for m in metrics:
    vals = mean_std[m]["values"]
    mean = mean_std[m]["mean"]
    std  = mean_std[m]["std"]
    vals_str = "".join(f"{v:<8}" for v in vals)
    print(f"{m:<25} {vals_str} {mean}±{std}")
print("="*70)

# ── Save full results ───────────────────────────────────────────────────────────
output = {"config": CONFIG, "per_run": all_runs, "mean_std": mean_std}
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=4, ensure_ascii=False)

print(f"\nFull results saved to '{OUTPUT_FILE}'.")
