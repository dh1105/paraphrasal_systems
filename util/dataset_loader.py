import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader
from tqdm import tqdm

class DatasetLoader(Dataset):

    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def create_dataset(self, batch_size):
        self.data.dropna(inplace=True)
        # if "t5" in self.tokenizer.name_or_path:
        self.data["sentence1"] = self.data["sentence1"].apply(lambda sen: "paraphrase: " + sen + "</s>")
        model_inputs = self.data["sentence1"]
        model_labels = self.data["sentence2"]

        input_encoding = self.tokenizer(
                [model_input for model_input in tqdm(model_inputs)],
                padding="longest",
                max_length=300 if self.tokenizer.name_or_path == "t5-3b" else 512,
                truncation=True,
                return_tensors="pt",
            )
        
        label_encoding = self.tokenizer(
                [model_label for model_label in tqdm(model_labels)],
                padding="longest",
                max_length=300 if self.tokenizer.name_or_path == "t5-3b" else 512,
                truncation=True,
                return_tensors="pt",
            )

        input_ids, attention_masks = input_encoding.input_ids, input_encoding.attention_mask
        labels = label_encoding.input_ids
        labels[labels == self.tokenizer.pad_token_id] = -100

        tensor_dataset = TensorDataset(input_ids, attention_masks, labels)

        data_loader = DataLoader(
            tensor_dataset,
            shuffle=True,
            batch_size=batch_size
        )

        return data_loader