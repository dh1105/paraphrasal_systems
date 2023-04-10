import pandas as pd
from sklearn.model_selection import train_test_split
import os

def read_mrpc_data(mrpc_data_path):
    mrpc_data = pd.read_csv(mrpc_data_path, sep="\t", on_bad_lines='skip')
    mrpc_data.rename(columns={"Quality": "label", "#1 String": "sentence1", "#2 String": "sentence2"}, inplace=True)
    mrpc_data["id"] = "s1_" + mrpc_data["#1 ID"].astype(str) + "_s2_" + mrpc_data["#2 ID"].astype(str)
    mrpc_data.drop(["#1 ID", "#2 ID"], axis=1, inplace=True)
    # print(mrpc_data.head())
    return filter_paraphrases_from_data(mrpc_data)

def read_qqp_or_paws_data(path):
    data = pd.read_csv(path, sep="\t")
    # print(data.head())
    return filter_paraphrases_from_data(data)

def filter_paraphrases_from_data(data):
    return data[data['label'] == 1]

def merge_data(mrpc_data, qq_data, paws_data):
    combined_data = pd.concat([mrpc_data, qq_data, paws_data])
    # print(combined_data.head())
    return combined_data

def create_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print ("Created a path: %s"%(path))

def save_data(data, path, name):
    create_path(path)
    data.to_csv(os.path.join(path, name), index=False)

if __name__ == '__main__':
    mrpc_save_path = "data/mrpc_final_data/"
    paws_qqp_save_path = "data/paws_qqp_final_data/"
    merged_data_save_path = "data/merged_final_data/"

    mrpc_test_data_path = "data/mrpc/msr_paraphrase_test.txt"
    mrpc_train_data_path = "data/mrpc/msr_paraphrase_train.txt"

    paws_dev_data_path = "data/paws_final/dev.tsv"
    paws_test_data_path = "data/paws_final/test.tsv"
    paws_train_data_path = "data/paws_final/train.tsv"

    paws_qqp_dev_test_data_path = "data/paws_qqp/output/dev_and_test.tsv"
    paws_qqp_train_data_path = "data/paws_qqp/output/train.tsv"

    dataset_names = ["dev.csv", "test.csv", "train.csv"]

    mrpc_test_data = read_mrpc_data(mrpc_test_data_path)
    mrpc_dev_data, mrpc_test_data = train_test_split(mrpc_test_data, test_size=0.5, random_state=42)
    mrpc_train_data = read_mrpc_data(mrpc_train_data_path)
    print("MRPC dev, test, train len:", len(mrpc_dev_data), len(mrpc_test_data), len(mrpc_train_data))

    mrpc_datasets = [mrpc_dev_data, mrpc_test_data, mrpc_train_data]

    for name, data in zip(dataset_names, mrpc_datasets):
        save_data(data, mrpc_save_path, name)

    paws_dev_data = read_qqp_or_paws_data(paws_dev_data_path)
    paws_test_data = read_qqp_or_paws_data(paws_test_data_path)
    paws_train_data = read_qqp_or_paws_data(paws_train_data_path)

    print("PAWS dev, test, train len:", len(paws_dev_data), len(paws_test_data), len(paws_train_data))

    paws_qqp_dev_test_data = read_qqp_or_paws_data(paws_qqp_dev_test_data_path)
    paws_qqp_dev_data, paws_qqp_test_data = train_test_split(paws_qqp_dev_test_data, test_size=0.5, random_state=42)
    paws_qqp_train_data = read_qqp_or_paws_data(paws_qqp_train_data_path)

    print("PAWS QQP dev, test, train len:", len(paws_qqp_dev_data), len(paws_qqp_test_data), len(paws_qqp_train_data))

    paws_qqp_datasets = [paws_qqp_dev_data, paws_qqp_test_data, paws_qqp_train_data]

    for name, data in zip(dataset_names, paws_qqp_datasets):
        save_data(data, paws_qqp_save_path, name)

    merged_dev_data = merge_data(mrpc_dev_data, paws_qqp_dev_data, paws_dev_data)
    merged_test_data = merge_data(mrpc_test_data, paws_qqp_test_data, paws_test_data)
    merged_train_data = merge_data(mrpc_train_data, paws_qqp_train_data, paws_train_data)

    merged_datasets = [merged_dev_data, merged_test_data, merged_train_data]

    for name, data in zip(dataset_names, merged_datasets):
        print(name, len(data))
        save_data(data, merged_data_save_path, name)