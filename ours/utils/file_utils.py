import json

def read_yaml(path):
    import yaml
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def write_json(data, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def append_jsonl(data, path):
    with open(path, 'a', encoding='utf-8') as f:
        for item in data:
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')

def read_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def read_xlsx(path):
    import pandas as pd
    df = pd.read_excel(path, engine="openpyxl")
    return df.to_dict(orient="records")  
