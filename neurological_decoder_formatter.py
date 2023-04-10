import pandas as pd
import os
import json

def save_data(sentences, path, name):
    with open(os.path.join(path, name), 'w') as file:
        for sentence in sentences:
            file.write(sentence + "\n")

def fetch_column(df, column_name, file_name):
    save_path = "/home/dhverma/neurologic_decoding/dataset/mnli"
    sentences = df[column_name].astype(str).to_list()
    sentences = [sentence.replace("`", "'") for sentence in sentences]
    save_data(sentences, save_path, file_name)

def read_jsonl_file(path):
    with open(path, 'r') as json_file:
        json_list = list(json_file)

    results = []
    for json_str in json_list:
        results.append(json.loads(json_str))
    return results

def load_data(path):
    return pd.DataFrame(read_jsonl_file(path))

base_path = "/home/dhverma/Determining-Robustness-of-NLU-Models/data/multinli_1.0/multinli_1.0_dev_mismatched.jsonl"
#"/home/dhverma/Determining-Robustness-of-NLU-Models/data/RTE_data/RTE_test.jsonl"
# /home/dhverma/Determining-Robustness-of-NLU-Models/data/multinli_1.0/multinli_1.0_dev_mismatched.jsonl
# dev_path = os.path.join(base_path, "dev.csv")
# test_path = os.path.join(base_path, "test.csv")
# train_path = os.path.join(base_path, "train.csv")

# dev_data = pd.read_csv(dev_path)
# test_data = pd.read_csv(test_path)
# train_data = pd.read_csv(train_path)
test_data = load_data(base_path)

# fetch_column(dev_data, "sentence1", "dev.source")
# fetch_column(dev_data, "sentence2", "dev.target")

fetch_column(test_data, "sentence1", "sentence1.source")
fetch_column(test_data, "sentence2", "sentence2.source")

# fetch_column(train_data, "sentence1", "train.source")
# fetch_column(train_data, "sentence2", "train.target")