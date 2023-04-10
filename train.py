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
from losses.jaccard_loss import JaccardLoss
from losses.cross_entropy_overlap_loss import CrossEntropyOverlapLoss
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import string

class Trainer:

    def __init__(self, options):
        self.model_name = options['model_name']
        self.device = options['device']
        self.train_path = options['train_path']
        self.val_path = options['val_path']
        self.batch_size = options['batch_size']
        self.epochs = options['epochs']
        self.save_path = options['save_path']
        self.loss_fn = options['loss_fn']
        transformer = Transformer(self.model_name)
        self.jaccard_loss = JaccardLoss()
        self.cross_entropy_overlap_loss = CrossEntropyOverlapLoss()
        self.model, self.tokenizer = transformer.get_model_and_tokenizer()
        self.model.to(self.device)

    def preprocess(self, a):
        table = str.maketrans(dict.fromkeys(string.punctuation))
        new_s = a.translate(table) 
        return new_s

    def compute_jaccard_index(self, s1, s2):
        s1 = self.preprocess(s1)
        s2 = self.preprocess(s2)
        
        s1 = set(s1.lower().split()) 
        s2 = set(s2.lower().split())
        intersection = s1.intersection(s2)

        union = s2.union(s1)
            
        # Calculate Jaccard similarity score 
        # using length of intersection set divided by length of union set
        return float(len(intersection)) / len(union)  

    def compute_avg_jaccard(self, inputs, paraphrases):
        avg_jacc = 0
        for input, paraphrase in zip(inputs, paraphrases):
            avg_jacc += self.compute_jaccard_index(input, paraphrase)
        
        return avg_jacc/len(inputs)

    def train(self, optimizer, scheduler, data_loader):
        self.model.train()
        self.model.zero_grad()
        total_loss = 0
        total_custom_loss = 0

        for batch_idx, (input_ids, attention_mask, labels) in enumerate(tqdm(data_loader)):
            padded_input_and_preds = pad_sequence([input_ids.transpose(0, 1), attention_mask.transpose(0, 1), labels.transpose(0, 1)], batch_first=True, padding_value=self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token))

            input_ids = padded_input_and_preds[0,:,:].transpose(0, 1).contiguous()
            attention_mask = padded_input_and_preds[1,:,:].transpose(0, 1).contiguous()
            labels = padded_input_and_preds[2,:,:].transpose(0, 1).contiguous()
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
            labels = labels.to(self.device)
            if 't5' in self.model_name:
                result = self.model(input_ids, attention_mask=attention_mask, labels=labels, return_dict=True)
                # result = torch.utils.checkpoint.checkpoint(self.model, input_ids, attention_mask, labels)
            else:
                result = self.model(input_ids, labels=labels, return_dict=True)
            logits = result['logits']
            # if batch_idx%100 == 0 and batch_idx > 0:
            #     softmax = nn.Softmax(dim=-1)
            #     logits = softmax(logits)
            #     argmax_preds = torch.argmax(logits, dim=-1)
            #     input = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            #     preds = self.tokenizer.batch_decode(argmax_preds, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            #     print(input)
            #     print(preds)

            if self.loss_fn == 'jaccard':
                custom_loss = self.jaccard_loss(input_ids, logits)
            elif self.loss_fn == 'overlap_ce':
                custom_loss = self.cross_entropy_overlap_loss(input_ids, logits)
            else:
                custom_loss = torch.zeros(1).to(self.device)

            loss = result['loss'] + custom_loss
            loss.backward()
            optimizer.step()
            self.model.zero_grad()
            scheduler.step()
            total_loss += result['loss'].item()
            total_custom_loss += custom_loss.item()
            # break

        loss = total_loss/len(data_loader)
        custom_loss = total_custom_loss/len(data_loader)
        return loss, custom_loss

    def validate(self, data_loader):
        self.model.eval()
        inputs = []
        paraphrase_1 = []
        paraphrase_2 = []
        paraphrase_3 = []

        with torch.no_grad():
            for batch_idx, (input_ids, attention_mask, labels) in enumerate(tqdm(data_loader)):
                # padded_input_and_preds = pad_sequence([input_ids.transpose(0, 1), attention_mask.transpose(0, 1), labels.transpose(0, 1)], batch_first=True)

                # input_ids = padded_input_and_preds[0,:,:].transpose(0, 1).contiguous()
                # attention_mask = padded_input_and_preds[1,:,:].transpose(0, 1).contiguous()
                # labels = padded_input_and_preds[2,:,:].transpose(0, 1).contiguous()
                # input_ids = input_ids.to(self.device)
                # attention_mask = attention_mask.to(self.device)
                # labels = labels.to(self.device)
                # result = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                # logits = result['logits']
                
                # if self.loss_fn == 'jaccard':
                #     custom_loss = self.jaccard_loss(input_ids, logits)
                # elif self.loss_fn == 'overlap_ce':
                #     custom_loss = self.cross_entropy_overlap_loss(input_ids, logits)
                # else:
                #     custom_loss = torch.zeros(1).to(self.device)

                # total_loss += result["loss"].item()
                # total_custom_loss += custom_loss.item()
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
                            num_return_sequences=3
                        )
                
                if "t5" in self.model_name:
                    labels[labels == -100] = 0
                
                input = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                inputs.extend(input)
                str_output = self.tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                str_output = [op.replace("Paraphrase:", "").strip() for op in str_output]
                str_output = [op.replace("paraphrase:", "").strip() for op in str_output]
                str_output = [op.replace("Paraphrase :", "").strip() for op in str_output]
                str_output = [op.replace("paraphrase :", "").strip() for op in str_output]
                for i in range(0, len(str_output), 3):
                    paraphrase_1.append(str_output[i])
                    paraphrase_2.append(str_output[i + 1])
                    paraphrase_3.append(str_output[i + 2])
        
        avg_para1_jacc, avg_para1_pos = model_utils.compute_avg_jaccard(inputs, paraphrase_1)
        avg_para2_jacc, avg_para2_pos = model_utils.compute_avg_jaccard(inputs, paraphrase_2)
        avg_para3_jacc, avg_para3_pos = model_utils.compute_avg_jaccard(inputs, paraphrase_3)

        # if avg_para1_jacc == avg_para2_jacc and avg_para2_jacc == avg_para3_jacc:
        #     min_jacc = min(avg_para1_pos, min(avg_para2_pos, avg_para3_pos))
        # else:
        #     min_jacc = min(avg_para1_jacc, min(avg_para2_jacc, avg_para3_jacc))
        min_jacc = min(avg_para1_jacc, min(avg_para2_jacc, avg_para3_jacc))
        return min_jacc

    def execute(self):
        total_t0 = time.time()
        last_best = 1.0
        print("Training model..")

        train_df = load_utils.load_data(self.train_path)
        val_df = load_utils.load_data(self.val_path)

        train_dataset = DatasetLoader(train_df, self.tokenizer)
        train_data_loader = train_dataset.create_dataset(self.batch_size)

        val_dataset = DatasetLoader(val_df, self.tokenizer)
        val_data_loader = val_dataset.create_dataset(self.batch_size)

        optimizer = AdamW(self.model.parameters(),
                lr = 1e-4,#lr = 4e-5, # args.learning_rate - default is 5e-5, for t5 it is 1e-4
                eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
            )

        total_steps = len(train_data_loader) * self.epochs

        scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                    num_warmup_steps = 1, # Default value in run_glue.py
                                                    num_training_steps = total_steps)

        for epoch_i in range(0, self.epochs):
            train_loss, train_jaccard_loss = self.train(optimizer, scheduler, train_data_loader)
            avg_jacc = self.validate(val_data_loader)

            print(f'Epoch {epoch_i + 1}: train_loss: {train_loss:.4f} train_custom_loss: {train_jaccard_loss:.4f} | avg_jacc: {avg_jacc:.4f}')
            # break
            # if val_loss > last_best:
                # print("Saving model..")
            if avg_jacc < last_best:
                print("Saving model..")
                last_best= avg_jacc
                model_utils.save_transformer(self.model, self.tokenizer, self.model_name, self.save_path)

        print("Training complete!")
        print("Total training took {:} (h:mm:ss)".format(model_utils.format_time(time.time()-total_t0)))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", help="Path to the training dataset csv file", default="./data/merged_final_data/train.csv")
    parser.add_argument("--val_path", help="Path to the validation dataset csv file", default="./data/merged_final_data/dev.csv")
    parser.add_argument("--save_path", help="Directory to save the model", default="./saved_model")
    parser.add_argument("--batch_size", help="Batch size", type=int, default=4)
    parser.add_argument("--epochs", help="Number of epochs", type=int, default=5)
    parser.add_argument("--loss_fn", help="The loss function you to use to use apart from the default model loss", choices=["jaccard", "overlap_ce", ""], default="")
    # parser.add_argument("--gradient_accumulation", help="Number of batches to accumulate gradients", type=int, default=0)
    parser.add_argument("--model_name", help="Name of the huggingface model or the path to the directory containing a pre-trained transformer", default="t5-large")
    return parser.parse_args()

def create_path(path):
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
    save_path = f'{args.save_path}/'
    create_path(save_path)
    
    options = {}
    options['batch_size'] = args.batch_size
    options['device'] = device
    options['train_path'] = args.train_path
    options['val_path'] = args.val_path
    options['model_name'] = args.model_name
    options['save_path'] = args.save_path
    options['epochs'] = args.epochs
    options['loss_fn'] = args.loss_fn
    # options['gradient_accumulation'] = args.gradient_accumulation
    print(options)

    trainer = Trainer(options)
    trainer.execute()