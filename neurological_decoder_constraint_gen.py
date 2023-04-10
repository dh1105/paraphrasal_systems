import pandas as pd
import os
import stanza
import json
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from tqdm import tqdm
import re
import string
from pyinflect import getAllInflections
import json

def save_data(sentences, path, name):
    with open(os.path.join(path, name), 'w') as file:
        for sentence in sentences:
            json.dump(sentence, file)
            file.write("\n")

def generate_positive_constraint(sentence, nlp):
    doc = nlp(sentence)
    # print(sentence)
    stop_words = set(stopwords.words('english'))

    for sent in doc.sentences:
        constraints = {}
        for word in sent.words:
            if word.text.lower() not in string.punctuation:
                valid_types = ["NOUN", "ADJ", "VERB", "ADV"]
                word_net_mapping = {
                    "NOUN": wordnet.NOUN,
                    "VERB": wordnet.VERB,
                    "ADJ": wordnet.ADJ,
                    "ADV": wordnet.ADV
                }
                if word.upos in valid_types:
                    # if word.upos != "INTJ":
                    #     inflect = getAllInflections(word.text, pos_type=word.upos[0])
                    # else:
                    #     inflect = getAllInflections(word.text)
                    inflect = getAllInflections(word.text, pos_type=word.upos[0])
                    
                    word_set = set()
                    for k,v in inflect.items():
                        set_v = set(v)
                        for word_v in set_v:
                            word_set.add(word_v)

                    syns = wordnet.synsets(word.text, word_net_mapping[word.upos], lang='eng')
                    # syn_set = set()
                    for syn in syns:
                        for i in syn.lemmas():
                            name = i.name()
                            # name = re.sub('[^0-9a-zA-Z]+', ' ', name)
                            name = name.replace("_", " ")
                            word_set.add(name)

                    word_set.discard(word.text)
                    if len(word_set) > 0:
                        constraints[word.text] = list(word_set)

    final_constaints = []
    for value in constraints.values():
        final_constaints.append(value)

    return final_constaints

def generate_negative_constraint(sentence, nlp):
    doc = nlp(sentence)
    stop_words = set(stopwords.words('english'))

    constraints = []
    for sent in doc.sentences:
        # constraints = {}
        for word in sent.words:
            if word.text.lower() not in stop_words and word.text.lower() not in string.punctuation:
                valid_types = ["NOUN", "ADJ", "VERB", "ADV"]
                if word.upos in valid_types:
                    # con = list(set([word.text, word.text.lower()]))
                    constraints.append(word.text)
                    constraints.append(word.text.lower())

    final_constaints = []
    for constraint in set(constraints):
        assert len([constraint]) == 1
        final_constaints.append([constraint])

    return final_constaints


def generate_constraint_and_save(sentences, nlp, name):
    save_path = "/home/dhverma/neurologic_decoding/dataset/mnli/constraint"
    # "/home/dhverma/Determining-Robustness-of-NLU-Models/data/multinli_1.0/multinli_1.0_dev_mismatched.jsonl"
    all_constraints = []
    for sentence in tqdm(sentences):
        sentence_constraint = generate_positive_constraint(sentence, nlp)
        # sentence_constraint = generate_negative_constraint(sentence, nlp)
        all_constraints.append(sentence_constraint)
    return save_data(all_constraints, save_path, name)

def read_jsonl_file(path):
    with open(path, 'r') as json_file:
        json_list = list(json_file)

    results = []
    for json_str in json_list:
        results.append(json.loads(json_str))
    return results

def load_data(path):
    return pd.DataFrame(read_jsonl_file(path))

# def fetch_column(df, column_name, file_name):
#     save_path = "/home/dhverma/neurologic_decoding/dataset/paraphrase"
#     sentences = df[column_name].astype(str).to_list()
#     save_data(sentences, save_path, file_name)

nlp = stanza.Pipeline(lang='en', processors='tokenize,pos,lemma')
base_path = "/home/dhverma/Determining-Robustness-of-NLU-Models/data/multinli_1.0/multinli_1.0_dev_mismatched.jsonl"
# dev_path = os.path.join(base_path, "dev.csv")
# test_path = os.path.join(base_path, "test.csv")
# train_path = os.path.join(base_path, "train.csv")

# dev_data = pd.read_csv(dev_path)
test_data = load_data(base_path)
# train_data = pd.read_csv(train_path)

# fetch_column(dev_data, "sentence1", "dev.source")
# fetch_column(dev_data, "sentence2", "dev.target")

# fetch_column(test_data, "sentence1", "test.source")
# fetch_column(test_data, "sentence2", "test.target")

# fetch_column(train_data, "sentence1", "train.source")
# fetch_column(train_data, "sentence2", "train.target")

# generate_constraint_and_save(dev_data["sentence1"].astype(str).to_list(), nlp, "dev.constraint.json")
generate_constraint_and_save(test_data["sentence1"].astype(str).to_list(), nlp, "sentence1.constraint.json")
generate_constraint_and_save(test_data["sentence2"].astype(str).to_list(), nlp, "sentence2.constraint.json")