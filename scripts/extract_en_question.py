import json
import sys
import os

def load_data(input_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def extract_data(data):
    results = []
    
    for item in data["questions"]:
        q_id = item["id"]
        
        question_en = None
        for q in item["question"]:
            if q.get("language") == "en":
                question_en = q.get("string")
                break
        
        if not question_en:
            continue
        
        sparql_gold = item.get("query", {}).get("sparql", "")
        
        results.append({
            "id": f"q{q_id}",
            "question": question_en,
            "sparql_gold": sparql_gold
        })
    
    return results

def save_data(output_path, data):
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 extract_en_question.py <input.json> <output.json>")
        print("Example: python3 extract_en_question.py input.json output.json")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    
    # Auto-add 'extracted_' prefix if output is just a directory or base name
    output_dir = os.path.dirname(output_path)
    output_filename = os.path.basename(output_path)
    
    if not output_filename.startswith('extracted_'):
        output_filename = 'extracted_' + output_filename
        output_path = os.path.join(output_dir, output_filename)
    
    data = load_data(input_path)
    extracted = extract_data(data)
    save_data(output_path, extracted)
    
    print(f"Saved {len(extracted)} examples to {output_path}")
