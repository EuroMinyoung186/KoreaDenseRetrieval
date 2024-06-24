import torch
import pickle

from typing import List, Tuple
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from torch import tensor as T


def wiki_collator(batch: List, padding_value: int = 1) -> Tuple[torch.Tensor]:
    """passage를 batch로 반환합니다."""
    idx = [e[0] for e in batch]
    batch_p = pad_sequence(
        [T(e[1]) for e in batch], batch_first=True, padding_value=padding_value
    )
    batch_p_attn_mask = (batch_p != padding_value).long()
    return (idx, batch_p, batch_p_attn_mask)

def custom_collate_fn(batch):
    return wiki_collator(batch, padding_value=1)

class WikiDatset(Dataset):
    def __init__(self, file_path = './all_wiki_data.pkl'):
        """
        file_path: pickle 파일 경로
        """
        self.data = self.load_data(file_path)

    def load_data(self, file_path):
        """
        pickle 파일에서 데이터를 로드합니다.
        """
        data = []
        with open(file_path, 'rb') as file:
            while True:
                try:
                    data.append(pickle.load(file))
                except EOFError:
                    break
        return data

    def __len__(self):
        """
        데이터셋의 총 길이를 반환합니다.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        인덱스에 해당하는 데이터를 반환합니다.
        """
        return self.data[idx]
    
