from torch.utils.data import Dataset
import pandas as pd

class TranslationDataset(Dataset):
    def __init__(
            self, 
            dataset_path, 
            t_tokenizer, 
            s_tokenizer, 
            tokenizer_args,
            is_tmp=False):
        
        self.data = pd.read_csv(dataset_path)
        if is_tmp:
            self.data = self.data.iloc[:1000] # for testing
        self.t_tokenizer = t_tokenizer
        self.s_tokenizer = s_tokenizer
        self.tokenizer_args = tokenizer_args

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ko = self.data.iloc[idx]['ko']
        en = self.data.iloc[idx]['en']

        t_en_inputs = tokenize(self.t_tokenizer, en, self.tokenizer_args)
        s_en_inputs = tokenize(self.s_tokenizer, en, self.tokenizer_args)
        s_ko_inputs = tokenize(self.s_tokenizer, ko, self.tokenizer_args)

        return t_en_inputs, s_en_inputs, s_ko_inputs
    

def tokenize(tokenizer, text, tokenizer_args):
    encoding = tokenizer(text, **tokenizer_args)
    input_ids = encoding['input_ids'].squeeze(0)
    attention_mask = encoding['attention_mask'].squeeze(0)
    return {'input_ids': input_ids, 'attention_mask': attention_mask}