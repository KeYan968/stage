"""
Replace QALD9plus static answers with fresh results from DBpedia endpoint.

Reads qald_9_plus_test_dbpedia.json, re-executes each gold SPARQL query
against the live DBpedia endpoint, and saves a new file with updated answers.

Usage:
    python3 update_qald_answers.py
"""

import json, time
from SPARQLWrapper import SPARQLWrapper, JSON, SPARQLExceptions

# ── Paths ──────────────────────────────────────────────────────────────────────
INPUT_FILE  = "/home/kyan/stage/Datasets/Raw_qald_plus_datasets/qald_9_plus_test_dbpedia.json"
OUTPUT_FILE = "/home/kyan/stage/Datasets/Raw_qald_plus_datasets/qald_9_plus_test_dbpedia_updated.json"
ENDPOINT    = "https://dbpedia.org/sparql"
TIMEOUT     = 15

# ── SPARQL execution ───────────────────────────────────────────────────────────
def execute_query(query):
    """Execute query on DBpedia, return QALD-format answers block."""
    sparql = SPARQLWrapper(ENDPOINT)
    sparql.setTimeout(TIMEOUT)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    try:
        qres     = sparql.query().convert()
        head     = qres.get("head", {"link": [], "vars": []})
        bindings = qres.get("results", {}).get("bindings", [])
        return {
            "status" : "success",
            "answers": [{
                "head"   : {"link": head.get("link", []), "vars": head.get("vars", [])},
                "results": {"distinct": False, "ordered": True, "bindings": bindings},
            }]
        }
    except SPARQLExceptions.QueryBadFormed as e:
        return {"status": "syntax_error", "answers": [{
            "head": {"link": [], "vars": []},
            "results": {"distinct": False, "ordered": True, "bindings": []},
        }]}
    except Exception as e:
        return {"status": "error", "answers": [{
            "head": {"link": [], "vars": []},
            "results": {"distinct": False, "ordered": True, "bindings": []},
        }]}

# ── Load QALD file ─────────────────────────────────────────────────────────────
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

questions    = data["questions"]
total        = len(questions)
success      = 0
syntax_error = 0
error        = 0

print(f"Updating answers for {total} questions...\n")

# ── Re-execute each gold query ─────────────────────────────────────────────────
for i, item in enumerate(questions):
    qid         = item.get("id", i)
    gold_query  = item.get("query", {}).get("sparql", "")

    result = execute_query(gold_query)
    status = result["status"]

    # Replace answers with fresh results
    item["answers"] = result["answers"]

    if status == "success":
        n_bindings = len(result["answers"][0]["results"]["bindings"])
        success += 1
        print(f"[{i+1}/{total}] id={qid} | {status} | {n_bindings} bindings")
    else:
        if status == "syntax_error":
            syntax_error += 1
        else:
            error += 1
        print(f"[{i+1}/{total}] id={qid} | {status}")

    time.sleep(0.3)  # polite rate limiting

# ── Save updated file ──────────────────────────────────────────────────────────
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=4, ensure_ascii=False)

print(f"""
Done.
  Total    : {total}
  Success  : {success}
  Syntax   : {syntax_error}
  Error    : {error}
Updated file saved to '{OUTPUT_FILE}'.
""")
