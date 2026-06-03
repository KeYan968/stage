#!/usr/bin/env python3
"""
python3 /home/kyan/stage/scripts/Qald_repair_creation/regeneration/extract_rows_for_regeneration.py\
  --raw_json /home/kyan/stage/Datasets/Raw_qald_plus_datasets/qald_9_plus_test_dbpedia.json\
  --id_csv /home/kyan/stage/Datasets/Regenerated_wrong_query/regeneration_test_ids.csv\
  --output_csv /home/kyan/stage/Datasets/Regenerated_wrong_query/syntax_regeneration_test_input.csv

python3 /home/kyan/stage/scripts/Qald_repair_creation/regeneration/extract_rows_for_regeneration.py\
  --raw_json /home/kyan/stage/Datasets/Raw_qald_plus_datasets/qald_9_plus_train_dbpedia.json\
  --id_csv /home/kyan/stage/Datasets/Regenerated_wrong_query/regeneration_training_ids.csv\
  --output_csv /home/kyan/stage/Datasets/Regenerated_wrong_query/syntax_regeneration_train_input.csv
"""

#!/usr/bin/env python3

import json
import pandas as pd
import argparse


def extract_question_text(question_field):

    if isinstance(question_field, list):

        for item in question_field:

            if item.get("language") == "en":
                return item.get("string", "")

        return question_field[0].get("string", "")

    return ""


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--raw_json",
        required=True
    )

    parser.add_argument(
        "--id_csv",
        required=True
    )

    parser.add_argument(
        "--output_csv",
        required=True
    )

    args = parser.parse_args()

    with open(args.raw_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    questions = data["questions"]

    rows = []

    for q in questions:

        rows.append({
            "id": "q" + str(q["id"]),
            "question": extract_question_text(
                q["question"]
            ),
            "sparql_gold": q["query"]["sparql"]
        })

    raw_df = pd.DataFrame(rows)

    ids_df = pd.read_csv(args.id_csv)

    ids = set(
        ids_df["id"].astype(str)
    )

    subset = raw_df[
        raw_df["id"].isin(ids)
    ]

    subset.to_csv(
        args.output_csv,
        index=False
    )

    print(
        f"Extracted {len(subset)} rows."
    )

    print(
        f"Saved to {args.output_csv}"
    )


if __name__ == "__main__":
    main()