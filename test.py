from models.transformer import Transformer
from tqdm import tqdm
import torch
import numpy as np
import pandas as pd
import argparse
import os
import time
import util.load_utils as load_utils
import util.model_utils as model_utils
from util.dataset_loader import DatasetLoader
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.nn.utils.rnn import pad_sequence
from losses.jaccard_loss import JaccardLoss
import torch.nn as nn
from evaluate import load
import string
from statistics import mean

class Tester:

    def __init__(self, options):
        self.model_name = options['model_name']
        self.device = options['device']
        self.test_path = options['test_path']
        self.batch_size = options['batch_size']
        self.inference_save_path = options['inference_save_path']
        transformer = Transformer(self.model_name)
        self.model, self.tokenizer = transformer.get_model_and_tokenizer()
        self.model.to(self.device)
        self.bertscore = load("bertscore")
        self.num_paraphrases = 3

    def compute_bert_score(self, preds, labels):
        results = self.bertscore.compute(predictions=preds, references=labels, lang="en")
        results["f1"] = mean(results["f1"])
        results.pop('hashcode', None)
        return results

    def test(self, data_loader):
        self.model.eval()

        inputs = []
        paraphrase_1 = []
        paraphrase_2 = []
        paraphrase_3 = []
        labels_list = []

        with torch.no_grad():
            for batch_idx, (input_ids, attention_mask, labels) in enumerate(tqdm(data_loader)):
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model.generate(
                            input_ids=input_ids, attention_mask=attention_mask,
                            max_length=256,
                            do_sample=True,
                            top_k=100,
                            top_p=0.95,
                            early_stopping=True,
                            num_return_sequences=self.num_paraphrases
                        )
                
                if "t5" in self.model_name:
                    labels[labels == -100] = 0
                input = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                label = self.tokenizer.batch_decode(labels, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                input = [ip.replace("paraphrase: ", "").strip() for ip in input]
                inputs.extend(input)
                labels_list.extend(label)
                # print(len(input))
                str_output = self.tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                str_output = [op.replace("Paraphrase:", "").strip() for op in str_output]
                str_output = [op.replace("paraphrase:", "").strip() for op in str_output]
                str_output = [op.replace("Paraphrase :", "").strip() for op in str_output]
                str_output = [op.replace("paraphrase :", "").strip() for op in str_output]
                # print(len(str_output))
                for i in range(0, len(str_output), self.num_paraphrases):
                    paraphrase_1.append(str_output[i])
                    paraphrase_2.append(str_output[i + 1])
                    paraphrase_3.append(str_output[i + 2])

        paraphrase_1_bert_score = self.compute_bert_score(paraphrase_1, labels_list)
        paraphrase_2_bert_score = self.compute_bert_score(paraphrase_2, labels_list)
        paraphrase_3_bert_score = self.compute_bert_score(paraphrase_3, labels_list)

        avg_para1_jacc, avg_para1_pos = model_utils.compute_avg_jaccard(inputs, paraphrase_1)
        avg_para2_jacc, avg_para2_pos = model_utils.compute_avg_jaccard(inputs, paraphrase_2)
        avg_para3_jacc, avg_para3_pos = model_utils.compute_avg_jaccard(inputs, paraphrase_3)

        metrics_dict = {
            "paraphrase1": {
                "bert_score": paraphrase_1_bert_score["f1"],
                "jaccard_score": avg_para1_jacc,
                "pos_score": avg_para1_pos
            },
            "paraphrase2": {
                "bert_score": paraphrase_2_bert_score["f1"],
                "jaccard_score": avg_para2_jacc,
                "pos_score": avg_para2_pos
            },
            "paraphrase3": {
                "bert_score": paraphrase_3_bert_score["f1"],
                "jaccard_score": avg_para3_jacc,
                "pos_score": avg_para3_pos
            }
        }

        inference_df = pd.DataFrame(columns=["sentence1", "sentence2", "paraphrase1", "paraphrase2", "paraphrase3"])
        inference_df["sentence1"] = inputs
        inference_df["sentence2"] = labels_list
        inference_df["paraphrase1"] = paraphrase_1
        inference_df["paraphrase2"] = paraphrase_2
        inference_df["paraphrase3"] = paraphrase_3


        return metrics_dict, inference_df

    def execute(self):
        total_t0 = time.time()
        print("Testing model..")

        test_df = load_utils.load_data(self.test_path)

        test_dataset = DatasetLoader(test_df, self.tokenizer)
        test_data_loader = test_dataset.create_dataset(self.batch_size)

        eval_metrics, inference_df = self.test(test_data_loader)
        print(eval_metrics)

        if self.inference_save_path is not None:
            inference_df.to_csv(self.inference_save_path, index=False)

        print("Testing complete!")
        print("Total testing took {:} (h:mm:ss)".format(model_utils.format_time(time.time()-total_t0)))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_path", help="Path to the testing dataset csv file", default="./data/merged_final_data/test.csv")
    parser.add_argument("--batch_size", help="Batch size", type=int, default=4)
    # parser.add_argument("--gradient_accumulation", help="Number of batches to accumulate gradients", type=int, default=0)
    parser.add_argument("--model_name", help="Name of the huggingface model or the path to the directory containing a pre-trained transformer", default="./saved_model/t5-large")
    parser.add_argument("--inference_save_path", help="Path to save the inference", default=None)
    return parser.parse_args()

def create_path(path):
    path = os.path.dirname(path)
    if not os.path.exists(path):
        os.makedirs(path)
        print ("Created a path: %s"%(path))

if __name__ == '__main__':
    # Set numpy, pytorch and python seeds for reproducibility.
    torch.manual_seed(42)
    np.random.seed(42)

    args = parse_args()
    # assert args.gradient_accumulation >= 0

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    create_path(args.inference_save_path)
    
    options = {}
    options['batch_size'] = args.batch_size
    options['device'] = device
    options['test_path'] = args.test_path
    options['model_name'] = args.model_name
    options['inference_save_path'] = args.inference_save_path
    # options['gradient_accumulation'] = args.gradient_accumulation
    print(options)

    tester = Tester(options)
    tester.execute()