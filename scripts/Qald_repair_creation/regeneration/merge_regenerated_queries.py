#!/usr/bin/env python3
"""
python3 /home/kyan/stage/scripts/Qald_repair_creation/regeneration/merge_regenerated_queries.py \
  --original_csv /home/kyan/stage/Datasets/Generated_wrong_sparql_datasets/test_set_with_subtypes.csv \
  --regen_csv /home/kyan/stage/Datasets/Regenerated_wrong_query/syntax_test_regenerated_with_subtypes.csv \
  --original_json /home/kyan/stage/Datasets/Generated_wrong_sparql_datasets/test_set_with_subtypes.json \
  --output_csv /home/kyan/stage/Datasets/Generated_wrong_sparql_datasets/test_set_updated.csv \
  --output_json /home/kyan/stage/Datasets/Generated_wrong_sparql_datasets/test_set_updated.json

python3 /home/kyan/stage/scripts/Qald_repair_creation/regeneration/merge_regenerated_queries.py \
  --original_csv /home/kyan/stage/Datasets/Generated_wrong_sparql_datasets/training_set_with_subtypes.csv \
  --regen_csv /home/kyan/stage/Datasets/Regenerated_wrong_query/syntax_train_regenerated_with_subtypes.csv \
  --original_json /home/kyan/stage/Datasets/Generated_wrong_sparql_datasets/training_set_with_subtypes.json \
  --output_csv /home/kyan/stage/Datasets/Generated_wrong_sparql_datasets/train_set_updated.csv \
  --output_json /home/kyan/stage/Datasets/Generated_wrong_sparql_datasets/train_set_updated.json
"""
import pandas as pd
import json
import ast
import argparse


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--original_csv", required=True)
    parser.add_argument("--regen_csv", required=True)
    parser.add_argument("--original_json", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--output_json", required=True)

    args = parser.parse_args()

    # Read original CSV
    original_csv = pd.read_csv(args.original_csv)

    # Read regenerated CSV
    regen_csv = pd.read_csv(args.regen_csv)

    # Build mapping
    regen_map = {}

    for idx in regen_csv.index:
        row_id = regen_csv.loc[idx, "id"]
        sub_error = regen_csv.loc[idx, "sub_error_types"]

        if isinstance(sub_error, str) and sub_error.startswith("["):
            try:
                sub_error = ast.literal_eval(sub_error)
            except:
                pass

        regen_map[row_id] = {
            "sparql_wrong": regen_csv.loc[idx, "sparql_wrong"],
            "sub_error_types": sub_error
        }

    # Update CSV
    for idx in original_csv.index:
        row_id = original_csv.loc[idx, "id"]

        if row_id in regen_map:
            original_csv.loc[idx, "sparql_wrong"] = regen_map[row_id]["sparql_wrong"]
            original_csv.loc[idx, "sub_error_types"] = str(regen_map[row_id]["sub_error_types"])

    original_csv.to_csv(args.output_csv, index=False, encoding="utf-8")

    # Load JSON
    with open(args.original_json, "r", encoding="utf-8") as f:
        original_json = json.load(f)

    # Update JSON
    for item in original_json:
        if item["id"] in regen_map:
            item["sparql_wrong"] = regen_map[item["id"]]["sparql_wrong"]
            item["sub_error_types"] = regen_map[item["id"]]["sub_error_types"]

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(original_json, f, indent=2, ensure_ascii=False)

    print("CSV and JSON files have been successfully updated!")


if __name__ == "__main__":
    main()