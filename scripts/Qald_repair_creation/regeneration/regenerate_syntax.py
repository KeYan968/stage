#!/usr/bin/env python3

"""
python3 /home/kyan/stage/scripts/Qald_repair_creation/regeneration/regenerate_syntax.py\
  --input_csv /home/kyan/stage/Datasets/Regenerated_wrong_query/syntax_regeneration_test_input.csv \
  --output_csv /home/kyan/stage/Datasets/Regenerated_wrong_query/syntax_test_regenerated.csv \
  --output_json /home/kyan/stage/Datasets/Regenerated_wrong_query/syntax_test_regenerated.json \
  --model gpt-4.1

python3 /home/kyan/stage/scripts/Qald_repair_creation/regeneration/regenerate_syntax.py \
  --input_csv /home/kyan/stage/Datasets/Regenerated_wrong_query/syntax_regeneration_train_input.csv \
  --output_csv /home/kyan/stage/Datasets/Regenerated_wrong_query/syntax_train_regenerated.csv \
  --output_json /home/kyan/stage/Datasets/Regenerated_wrong_query/syntax_train_regenerated.json \
  --model gpt-4.1
"""


#!/usr/bin/env python3

import argparse
import json
import os
import time
from typing import Dict, Tuple

import pandas as pd
from tqdm import tqdm
from openai import OpenAI


SYNTAX_PROMPT = """
You are an expert in SPARQL and DBpedia.

Your task is to generate ONE intentionally incorrect SPARQL query from a correct golden SPARQL query.

Syntactic errors are violations of SPARQL grammar rules that prevent the query from being parsed correctly.

Examples of syntactic errors include:
- missing or mismatched braces
- missing parentheses
- malformed SELECT clause
- incorrect WHERE placement
- malformed PREFIX declaration
- incorrect clause ordering
- undeclared variable
- general SPARQL grammar violations

Rules:
- Introduce EXACTLY ONE syntactic error
- Do NOT introduce semantic errors
- Do NOT introduce structural errors
- Keep the query as close as possible to the original
- MUST derive from gold query with minimal modification
- MUST NOT be identical to gold query
- Return ONLY valid JSON
- Do NOT include explanations or comments

Input:
__INPUT__

Output format:
__OUTPUT__
"""


def build_prompt(row: Dict) -> str:

    input_block = (
        "{\n"
        f'  "id": "{row["id"]}",\n'
        f'  "question": "{row["question"]}",\n'
        f'  "sparql_gold": "{row["sparql_gold"]}"\n'
        "}"
    )

    output_block = (
        "{\n"
        f'  "id": "{row["id"]}",\n'
        f'  "question": "{row["question"]}",\n'
        '  "sparql_wrong": "<generated incorrect SPARQL query>",\n'
        f'  "sparql_gold": "{row["sparql_gold"]}",\n'
        '  "error_types": ["syntax"]\n'
        "}"
    )

    return SYNTAX_PROMPT.replace("__INPUT__", input_block).replace("__OUTPUT__", output_block)


MAX_RETRY = 5


def call_model(client: OpenAI, model: str, prompt: str) -> Tuple[Dict, str]:

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,  # 🔥 stability
    )

    raw_text = response.choices[0].message.content.strip()

    if "```json" in raw_text:
        raw_text = raw_text.split("```json")[1].split("```")[0].strip()
    elif "```" in raw_text:
        raw_text = raw_text.split("```")[1].split("```")[0].strip()

    parsed_json = json.loads(raw_text)

    return parsed_json, raw_text


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--model", default="gpt-4.1")
    parser.add_argument("--sleep", type=float, default=0.5)

    args = parser.parse_args()

    if "OPENAI_API_KEY" not in os.environ:
        raise RuntimeError("OPENAI_API_KEY not found")

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)

    df = pd.read_csv(args.input_csv)

    required_columns = {"id", "question", "sparql_gold"}
    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    client = OpenAI()

    prompt_list = []
    sparql_wrong_list = []
    raw_json_list = []
    valid_generation_list = []
    issue_list = []

    for _, row in tqdm(df.iterrows(), total=len(df)):

        prompt = build_prompt(row)
        prompt_list.append(prompt)

        success = False
        final_wrong = ""
        final_raw = ""

        for _ in range(MAX_RETRY):

            try:
                parsed, raw_text = call_model(client, args.model, prompt)

                sparql_wrong = parsed.get("sparql_wrong", "").strip()

                # strict validation
                if (
                    sparql_wrong
                    and sparql_wrong.strip() != str(row["sparql_gold"]).strip()
                ):
                    success = True
                    final_wrong = sparql_wrong
                    final_raw = raw_text
                    break

            except Exception:
                pass

        if success:
            sparql_wrong_list.append(final_wrong)
            raw_json_list.append(final_raw)
            valid_generation_list.append("yes")
            issue_list.append("none")
        else:
            sparql_wrong_list.append("")
            raw_json_list.append("")
            valid_generation_list.append("no")
            issue_list.append("failed_or_identical")

        time.sleep(args.sleep)

    df["sparql_wrong"] = sparql_wrong_list
    df["error_types"] = [["syntax"] for _ in range(len(df))]
    df["prompt"] = prompt_list
    df["raw_json"] = raw_json_list
    df["valid_generation"] = valid_generation_list
    df["issue"] = issue_list

    df.to_csv(args.output_csv, index=False)

    df[
        ["id", "question", "sparql_wrong", "sparql_gold", "error_types"]
    ].to_json(
        args.output_json,
        orient="records",
        force_ascii=False,
        indent=2
    )

    print(f"Saved CSV: {args.output_csv}")
    print(f"Saved JSON: {args.output_json}")


if __name__ == "__main__":
    main()