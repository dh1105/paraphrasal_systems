from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import stanza
from pyinflect import getAllInflections, getInflection
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import wordnet
import string
from itertools import product
import numpy as np
import re
from losses.jaccard_loss import JaccardLoss

torch.manual_seed(42)

# def jaccard_similarity(s1, s2): 
#     # s1 = preprocess(s1)
#     # s2 = preprocess(s2)
    
#     # s1 = set(s1.lower().split()) 
#     # s2 = set(s2.lower().split())
#     # intersection = s1.intersection(s2)

#     # union = s2.union(s1)
        
#     # # Calculate Jaccard similarity score 
#     # # using length of intersection set divided by length of union set
#     # return float(len(intersection)) / len(union)
#     # jaccs = []
#     alpha = 0.5
#     beta = 0.5
#     s1 = preprocess(s1)
#     s2 = preprocess(s2)
    
#     s1_list = s1.lower().split(" ")
#     s2_list = s2.lower().split(" ")
#     s1 = set(s1_list) 
#     s2 = set(s2_list)
#     intersection = s1.intersection(s2)
#     union = s1.union(s2)

#     jaccard_score = alpha * float(len(intersection)) / len(union)

#     div = max(len(s1_list), len(s2_list))
#     if len(intersection) != 0:
#         pos_score = beta * sum([abs(s1_list.index(inter) - s2_list.index(inter))/div for inter in intersection]) /len(intersection)
#     else:
#         pos_score = 0
        
#     # Calculate Jaccard similarity score 
#     # using length of intersection set divided by length of union set
#     # jaccs.append(jaccard_score + pos_score)
#     return jaccard_score, pos_score, jaccard_score + pos_score
    
# def preprocess(a):
#     table = str.maketrans(dict.fromkeys(string.punctuation))
#     new_s = a.translate(table) 
#     return new_s

# tokenizer = AutoTokenizer.from_pretrained("saved_models/experiment_jl/t5-large")  
# model = AutoModelForSeq2SeqLM.from_pretrained("saved_models/experiment_jl/t5-large")

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# sentence = "They were there to enjoy us and they were there to pray for us."

# # text = "paraphrase: " + sentence + " </s>"
# text = sentence

# encoding = tokenizer.encode_plus(text,pad_to_max_length=True, return_tensors="pt")
# []
# input_ids, attention_masks = encoding["input_ids"].to(device), encoding["attention_mask"].to(device)
# model.to(device)

# outputs = model.generate(
#     input_ids=input_ids, attention_mask=attention_masks,
#     max_length=256,
#     do_sample=True,
#     top_k=100,
#     top_p=0.95,
#     early_stopping=True,
#     temperature=0.80,
#     num_return_sequences=10
# )

# for output in outputs:
#     line = tokenizer.decode(output, skip_special_tokens=True,clean_up_tokenization_spaces=True)
#     print(line, jaccard_similarity(sentence, line))

# jaccard_loss = JaccardLoss()

# input_ids = torch.tensor([1, 1, 2, 3])
# logits = torch.tensor([[0.1, 0.9, 0.3, 0.2],
#                        [0.2, 0.9, 0.3, 0.2],
#                        [0.2, 0.3, 0.9, 0.1],
#                        [0.1, 0.2, 0.3, 0.9]])

# custom_loss = jaccard_loss(input_ids, logits)
# print(custom_loss)

sent1 = "Bill gave a raise."
sent2 = "A raise was given by Bill."

import stanza
from pyinflect import getAllInflections
from nltk.corpus import stopwords
from nltk.corpus import wordnet

# from pattern.en import conjugate, lemma, lexeme, PRESENT, SG
# print (lemma('she'))
# print (lexeme('she'))
# print (conjugate(verb='give',tense=PRESENT,number=SG)) # he / she / it

nlp = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos,lemma,depparse')
doc = nlp(sent1)

print(*[f'word: {word.text} \tupos: {word.upos} \tdeprel: {word.deprel}' for sent in doc.sentences for word in sent.words], sep='\n')
print()
doc = nlp(sent2)

print(*[f'word: {word.text} \tupos: {word.upos} \tdeprel: {word.deprel}' for sent in doc.sentences for word in sent.words], sep='\n')

print()

inflect = getAllInflections("give", pos_type="V")
print(inflect)

