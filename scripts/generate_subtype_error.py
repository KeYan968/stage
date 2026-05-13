# classify_sparql_sub_error.py

#!/usr/bin/env python3
"""
Classify SPARQL error subtypes using the OpenAI API.

Inputs:
    1. CSV file produced by generate_wrong_sparql_query.py
    2. JSON file produced by generate_wrong_sparql_query.py

Outputs:
    1. Updated CSV with a new column: sub_error_types
    2. Updated JSON with a new field: sub_error_types

Usage:
    python3 generate_subtype_error.py \
        --input_csv /home/kyan/stage/Datasets/Generated_wrong_sparql_datasets/training_set_with_errors.csv \
        --input_json /home/kyan/stage/Datasets/Generated_wrong_sparql_datasets/training_set_with_errors.json \
        --output_csv /home/kyan/stage/Datasets/Generated_wrong_sparql_datasets/training_set_with_subtypes.csv \
        --output_json /home/kyan/stage/Datasets/Generated_wrong_sparql_datasets/training_set_with_subtypes.json \
        --model gpt-4.1

    python3 generate_subtype_error.py \
        --input_csv /home/kyan/stage/Datasets/Generated_wrong_sparql_datasets/test_set_with_errors.csv \
        --input_json /home/kyan/stage/Datasets/Generated_wrong_sparql_datasets/test_set_with_errors.json \
        --output_csv /home/kyan/stage/Datasets/Generated_wrong_sparql_datasets/test_set_with_subtypes.csv \
        --output_json /home/kyan/stage/Datasets/Generated_wrong_sparql_datasets/test_set_with_subtypes.json \
        --model gpt-4.1
"""

import argparse
import ast
import json
import os
import time
from typing import Dict, List, Tuple

import pandas as pd
from tqdm import tqdm
from openai import OpenAI


CLASSIFICATION_PROMPT = """
You are an expert in SPARQL and DBpedia.

Your task is to identify the most specific subtype of the error in the incorrect SPARQL query.

Possible subtypes:

Semantic error subtypes:
- incorrect URI
- incorrect property
- incorrect class
- hallucinated entity or property
- wrong literal or resource type
- malformed FILTER condition
- logically inconsistent triple patterns
- ontology constraint violations

Syntax error subtypes:
- missing or mismatched braces
- missing parentheses
- malformed SELECT clause
- incorrect WHERE placement
- malformed PREFIX declaration
- incorrect clause ordering
- undeclared variable
- general SPARQL grammar violations

Structural error subtypes:
- triple flip (subject/object inversion)
- incorrect joins between triples
- disconnected graph patterns
- missing triple components
- extra triple components
- malformed triple composition

Rules:
- Return EXACTLY ONE subtype.
- The subtype MUST be one of the options listed above.
- Return ONLY valid JSON.
- Do NOT include explanations.

Input:
{{
  "id": "{id}",
  "question": "{question}",
  "sparql_gold": "{sparql_gold}",
  "sparql_wrong": "{sparql_wrong}",
  "error_types": {error_types}
}}

Output format:
{{
  "id": "{id}",
  "sub_error_types": ["<one subtype from the list above>"]
}}
""".strip()


VALID_SUBTYPES = {
    "incorrect URI",
    "incorrect property",
    "incorrect class",
    "hallucinated entity or property",
    "wrong literal or resource type",
    "malformed FILTER condition",
    "logically inconsistent triple patterns",
    "ontology constraint violations",
    "missing or mismatched braces",
    "missing parentheses",
    "malformed SELECT clause",
    "incorrect WHERE placement",
    "malformed PREFIX declaration",
    "incorrect clause ordering",
    "undeclared variable",
    "general SPARQL grammar violations",
    "triple flip (subject/object inversion)",
    "incorrect joins between triples",
    "disconnected graph patterns",
    "missing triple components",
    "extra triple components",
    "malformed triple composition",
}


DEFAULT_SUBTYPE = {
    "semantic": "incorrect property",
    "syntax": "general SPARQL grammar violations",
    "structural": "triple flip (subject/object inversion)",
}


def escape_braces(text: str) -> str:
    """Escape braces so strings are safe in str.format()."""
    return text.replace("{", "{{").replace("}", "}}")


def parse_error_types(value) -> List[str]:
    """Convert CSV cell content to a Python list."""
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
            return [str(x) for x in parsed]
    except Exception:
        pass

    return [text]


def call_model(client: OpenAI, model: str, prompt: str) -> Tuple[Dict, str]:
    """Call the OpenAI model and parse the JSON response."""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.0,
    )

    raw_text = response.choices[0].message.content.strip()

    if "```json" in raw_text:
        raw_text = raw_text.split("```json", 1)[1].split("```", 1)[0].strip()
    elif raw_text.startswith("```"):
        raw_text = raw_text.split("```", 2)[1].strip()

    parsed_json = json.loads(raw_text)

    return parsed_json, raw_text


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--input_json", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--model", default="gpt-4.1")
    parser.add_argument("--sleep", type=float, default=0.2)

    args = parser.parse_args()

    if "OPENAI_API_KEY" not in os.environ:
        raise RuntimeError("Please set the OPENAI_API_KEY environment variable.")

    for path in [args.output_csv, args.output_json]:
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)

    df = pd.read_csv(args.input_csv)

    with open(args.input_json, "r", encoding="utf-8") as f:
        json_data = json.load(f)

    required_columns = {
        "id",
        "question",
        "sparql_gold",
        "sparql_wrong",
        "error_types",
    }

    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in CSV: {sorted(missing)}")

    client = OpenAI()

    sub_error_types_column = []
    classification_raw_column = []
    classification_issue_column = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        error_types = parse_error_types(row["error_types"])
        main_error_type = error_types[0] if error_types else "semantic"

        prompt = CLASSIFICATION_PROMPT.format(
            id=escape_braces(str(row["id"])),
            question=escape_braces(str(row["question"])),
            sparql_gold=escape_braces(str(row["sparql_gold"])),
            sparql_wrong=escape_braces(str(row["sparql_wrong"])),
            error_types=json.dumps(error_types, ensure_ascii=False),
        )

        try:
            parsed, raw_text = call_model(
                client=client,
                model=args.model,
                prompt=prompt,
            )

            subtype_list = parsed.get("sub_error_types", [])

            if not isinstance(subtype_list, list) or len(subtype_list) != 1:
                raise ValueError("Invalid sub_error_types format")


            subtype = str(subtype_list[0]).strip()

            if subtype not in VALID_SUBTYPES:
                raise ValueError(f"Invalid subtype: {subtype}")

            sub_error_types_column.append([subtype])
            classification_raw_column.append(raw_text)
            classification_issue_column.append("none")

        except Exception as e:
            fallback = DEFAULT_SUBTYPE.get(
                main_error_type,
                "general SPARQL grammar violations",
            )

            print(f"Error on id={row['id']}: {e}")

            sub_error_types_column.append([fallback])
            classification_raw_column.append("")
            classification_issue_column.append(str(e))

        time.sleep(args.sleep)

    df["sub_error_types"] = sub_error_types_column
    df["classification_raw_json"] = classification_raw_column
    df["classification_issue"] = classification_issue_column

    insert_position = (
        df.columns.get_loc("error_types") + 1
        if "error_types" in df.columns
        else len(df.columns)
    )

    cols = list(df.columns)

    if cols[-3] == "sub_error_types":  # remove from end and reinsert
        for col in [
            "sub_error_types",
            "classification_raw_json",
            "classification_issue",
        ]:
            cols.remove(col)

        cols.insert(insert_position, "sub_error_types")
        cols.append("classification_raw_json")
        cols.append("classification_issue")

        df = df[cols]

    df.to_csv(args.output_csv, index=False)

    subtype_by_id = {
        str(row["id"]): subtype
        for _, row, subtype in zip(
            range(len(df)),
            df.to_dict("records"),
            sub_error_types_column,
        )
    }

    for item in json_data:
        item_id = str(item.get("id", ""))
        item["sub_error_types"] = subtype_by_id.get(item_id, [])

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)

    print(f"Saved CSV to {args.output_csv}")
    print(f"Saved JSON to {args.output_json}")


if __name__ == "__main__":
    main()




