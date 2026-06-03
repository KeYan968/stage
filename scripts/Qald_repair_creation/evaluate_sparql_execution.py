#!/usr/bin/env python3
"""
Validate generated SPARQL queries by executing both the gold and wrong
queries against the DBpedia SPARQL endpoint.

Input:
    CSV file produced by generate_subtype_error.py
    (or any CSV containing at least:
        - id
        - sparql_gold
        - sparql_wrong
        - error_types
        - sub_error_types (optional)
    )

Output:
    Updated CSV with automatic validation fields.

Usage:
    python3 evaluate_sparql_execution.py \
        --input_csv /home/kyan/stage/Datasets/Generated_wrong_sparql_datasets/training_set_with_subtypes.csv \
        --output_csv /home/kyan/stage/Datasets/Validated_sparql_execuetion_records/training_set_validated.csv \
        --sleep 1.0

    python3 evaluate_sparql_execution.py \
        --input_csv /home/kyan/stage/Datasets/Generated_wrong_sparql_datasets/test_set_with_subtypes.csv \
        --output_csv /home/kyan/stage/Datasets/Validated_sparql_execuetion_records/test_set_validated.csv \
        --sleep 1.0

        python3 /home/kyan/stage/scripts/Qald_repair_creation/evaluate_sparql_execution.py \
        --input_csv /home/kyan/stage/Datasets/Regenerated_wrong_query/syntax_test_regenerated_with_subtypes.csv \
        --output_csv /home/kyan/stage/Datasets/Regenerated_wrong_query/validated_syntax_test_regenerated_with_subtypes.csv \
        --sleep 1.0

        python3 /home/kyan/stage/scripts/Qald_repair_creation/evaluate_sparql_execution.py \
        --input_csv /home/kyan/stage/Datasets/Regenerated_wrong_query/syntax_train_regenerated_with_subtypes.csv \
        --output_csv /home/kyan/stage/Datasets/Regenerated_wrong_query/validated_syntax_train_regenerated_with_subtypes.csv \
        --sleep 1.0
"""

import argparse
import ast
import time

import pandas as pd
import requests
from tqdm import tqdm

ENDPOINT = "https://dbpedia.org/sparql"


def normalize_query(query: str) -> str:
    """Normalize a query string."""
    if pd.isna(query):
        return ""
    return str(query).strip()


def execute_sparql(query: str, timeout=30):
    """
    Execute a SPARQL query against DBpedia.

    Returns:
        executes (bool): whether the query executed successfully
        result_count (int): number of returned bindings
        results_set (set): normalized set of tuples for comparison
        error_message (str): error message if execution failed
    """
    if not query.strip():
        return False, 0, set(), "empty_query"

    params = {
        "query": query,
        "format": "application/sparql-results+json",
    }

    try:
        response = requests.get(
            ENDPOINT,
            params=params,
            timeout=timeout,
            headers={"Accept": "application/sparql-results+json"},
        )

        response.raise_for_status()

        data = response.json()
        bindings = data.get("results", {}).get("bindings", [])

        normalized_results = set()

        for binding in bindings:
            row = tuple(
                sorted(
                    (var, binding[var].get("value", ""))
                    for var in binding
                )
            )
            normalized_results.add(row)

        return True, len(bindings), normalized_results, "none"

    except requests.exceptions.Timeout:
        return False, 0, set(), "timeout"

    except requests.exceptions.HTTPError as e:
        return False, 0, set(), f"http_error: {e}"

    except requests.exceptions.RequestException as e:
        return False, 0, set(), f"request_error: {e}"

    except Exception as e:
        return False, 0, set(), str(e)


def parse_list_column(value):
    """Convert a CSV cell containing a Python list into a list."""
    if isinstance(value, list):
        return value

    if pd.isna(value):
        return []

    text = str(value).strip()

    if not text:
        return []

    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            return parsed
    except Exception:
        pass

    return [text]


def classify_behavior(
    row,
    gold_exec_ok,
    wrong_exec_ok,
    gold_count,
    wrong_count,
    same_results,
):
    """
    Assign a single validation category.

    This category combines:
    - execution status
    - result comparison
    - suspicious cases requiring manual review
    """
    gold_query = normalize_query(row["sparql_gold"])
    wrong_query = normalize_query(row["sparql_wrong"])

    error_types = parse_list_column(row["error_types"])
    main_error_type = error_types[0] if error_types else ""

    # Empty or identical queries
    if not wrong_query:
        return "wrong_query_empty"

    if gold_query == wrong_query:
        return "wrong_identical_to_gold"

    # Execution failures
    if not gold_exec_ok:
        return "gold_execution_failed"

    if not wrong_exec_ok:
        if main_error_type != "syntax":
            return "non_syntax_but_fails"
        return "wrong_execution_failed"

    # Syntax errors that still execute
    if main_error_type == "syntax":
        return "syntax_but_executes"

    # Same results
    if same_results:
        if main_error_type in {"semantic", "structural"}:
            return "possible_ineffective_corruption"
        return "same_results"

    # Result comparison
    if gold_count > 0 and wrong_count == 0:
        return "gold_has_results_wrong_no_results"

    if gold_count == 0 and wrong_count == 0:
        return "both_no_results"

    if gold_count == 0 and wrong_count > 0:
        return "gold_no_results_wrong_has_results"

    return "different_results"


def determine_manual_review_needed(behavior_category):
    """
    Decide whether this instance should be manually reviewed.
    """
    categories_requiring_review = {
        "wrong_query_empty",
        "wrong_identical_to_gold",
        "gold_execution_failed",
        "wrong_execution_failed",
        "syntax_but_executes",
        "non_syntax_but_fails",
        "same_results",
        "possible_ineffective_corruption",
    }

    if behavior_category in categories_requiring_review:
        return "yes"
    return "no"


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_csv",
        required=True,
        help="Input CSV file containing SPARQL queries.",
    )

    parser.add_argument(
        "--output_csv",
        required=True,
        help="Output CSV file with validation results.",
    )

    parser.add_argument(
        "--sleep",
        type=float,
        default=1.0,
        help="Sleep time between endpoint calls (seconds).",
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="Timeout for each SPARQL query in seconds.",
    )

    args = parser.parse_args()

    # Load data
    df = pd.read_csv(args.input_csv)

    # Validate required columns
    required_columns = {
        "id",
        "sparql_gold",
        "sparql_wrong",
        "error_types",
    }

    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # Output columns
    gold_exec_ok_col = []
    gold_result_count_col = []
    gold_error_col = []

    wrong_exec_ok_col = []
    wrong_result_count_col = []
    wrong_error_col = []

    same_results_col = []
    behavior_category_col = []
    manual_review_needed_col = []
    manual_review_notes_col = []

    # Process each row
    for _, row in tqdm(df.iterrows(), total=len(df)):
        gold_query = normalize_query(row["sparql_gold"])
        wrong_query = normalize_query(row["sparql_wrong"])

        # Execute gold query
        gold_exec_ok, gold_count, gold_results, gold_error = execute_sparql(
            gold_query,
            timeout=args.timeout,
        )

        time.sleep(args.sleep)

        # Execute wrong query
        wrong_exec_ok, wrong_count, wrong_results, wrong_error = execute_sparql(
            wrong_query,
            timeout=args.timeout,
        )

        time.sleep(args.sleep)

        # Compare results
        same_results = (
            gold_exec_ok
            and wrong_exec_ok
            and gold_results == wrong_results
        )

        # Single validation category
        behavior_category = classify_behavior(
            row=row,
            gold_exec_ok=gold_exec_ok,
            wrong_exec_ok=wrong_exec_ok,
            gold_count=gold_count,
            wrong_count=wrong_count,
            same_results=same_results,
        )

        # Manual review flag
        manual_review_needed = determine_manual_review_needed(
            behavior_category
        )

        # Save values
        gold_exec_ok_col.append(gold_exec_ok)
        gold_result_count_col.append(gold_count)
        gold_error_col.append(gold_error)

        wrong_exec_ok_col.append(wrong_exec_ok)
        wrong_result_count_col.append(wrong_count)
        wrong_error_col.append(wrong_error)

        same_results_col.append(same_results)
        behavior_category_col.append(behavior_category)
        manual_review_needed_col.append(manual_review_needed)
        manual_review_notes_col.append("")

    # Add validation columns
    df["gold_exec_ok"] = gold_exec_ok_col
    df["gold_result_count"] = gold_result_count_col
    df["gold_error"] = gold_error_col

    df["wrong_exec_ok"] = wrong_exec_ok_col
    df["wrong_result_count"] = wrong_result_count_col
    df["wrong_error"] = wrong_error_col

    df["same_results"] = same_results_col
    df["behavior_category"] = behavior_category_col
    df["manual_review_needed"] = manual_review_needed_col
    df["manual_review_notes"] = manual_review_notes_col

    # Save CSV
    df.to_csv(args.output_csv, index=False)

    print(f"Saved validation results to {args.output_csv}")

    # Print summary
    print("\nBehavior summary:")
    print(df["behavior_category"].value_counts())

    print("\nManual review summary:")
    print(df["manual_review_needed"].value_counts())


if __name__ == "__main__":
    main()