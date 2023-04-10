import os
import datetime
import string

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

def save_transformer(model, tokenizer, model_name, path):
    full_path = os.path.join(path, model_name)
    model.save_pretrained(full_path)
    tokenizer.save_pretrained(full_path)

def compute_avg_jaccard(inputs, paraphrases):
    avg_jacc = 0
    avg_pos = 0
    for input, paraphrase in zip(inputs, paraphrases):
        jaccard, pos = compute_jaccard_index(input, paraphrase)
        avg_jacc += jaccard
        avg_pos += pos
    
    return avg_jacc/len(inputs), avg_pos/len(inputs)

def preprocess(a):
    table = str.maketrans(dict.fromkeys(string.punctuation))
    new_s = a.translate(table) 
    return new_s

def compute_jaccard_index(s1, s2):
    s1 = preprocess(s1)
    s2 = preprocess(s2)
    
    s1_list = s1.lower().split()
    s2_list = s2.lower().split()

    s1 = set(s1.lower().split()) 
    s2 = set(s2.lower().split())
    intersection = s1.intersection(s2)

    union = s2.union(s1)

    jaccard_score = float(len(intersection)) / len(union)

    div = max(len(s1_list), len(s2_list))
    if len(intersection) != 0:
        pos_score = sum([abs(s1_list.index(inter) - s2_list.index(inter))/div for inter in intersection]) /len(intersection)
    else:
        pos_score = 0
        
    # Calculate Jaccard similarity score 
    # using length of intersection set divided by length of union set
    return jaccard_score, pos_score