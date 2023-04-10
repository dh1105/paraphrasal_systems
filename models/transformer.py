from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
# from transformers import T5Tokenizer, T5ForConditionalGeneration

class Transformer():

    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        print(type(self.model))

    def get_model_and_tokenizer(self):
        return self.model, self.tokenizer