#!/usr/bin/env python3

import pandas as pd
import argparse


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--validated_csv",
        required=True,
        help="Validated CSV file"
    )

    parser.add_argument(
        "--output_ids",
        required=True,
        help="Output CSV containing ids to regenerate"
    )

    args = parser.parse_args()

    df = pd.read_csv(args.validated_csv)

    subset = df[
        df["requires_regeneration"]
          .astype(str)
          .str.lower()
          .eq("yes")
    ]

    ids_df = subset[["id"]].drop_duplicates()

    ids_df.to_csv(
        args.output_ids,
        index=False
    )

    print(f"Found {len(ids_df)} ids requiring regeneration.")
    print(f"Saved to {args.output_ids}")


if __name__ == "__main__":
    main()
"""
python3 /home/kyan/stage/scripts/Qald_repair_creation/regeneration/extract_regeneration_ids.py \
  --validated_csv /home/kyan/stage/Datasets/Validated_sparql_execution_records/test_set_validated_manually.csv \
  --output_ids /home/kyan/stage/Datasets/Regenerated_wrong_query/regeneration_test_ids.csv

python3 /home/kyan/stage/scripts/Qald_repair_creation/regeneration/extract_regeneration_ids.py \
  --validated_csv /home/kyan/stage/Datasets/Validated_sparql_execution_records/training_set_validated_manually.csv \
  --output_ids /home/kyan/stage/Datasets/Regenerated_wrong_query/regeneration_training_ids.csv
"""
