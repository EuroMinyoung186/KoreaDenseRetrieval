from tqdm import tqdm
import torch
from torch import tensor as T
from torch.nn.utils.rnn import pad_sequence
import os
import json
import re
from typing import Iterator, List, Sized, Tuple
from transformers import AutoTokenizer
import pickle
from glob import glob
import torch

from utils_e import get_passage_file

# se


def xorqa_collator(batch: List[Tuple], padding_value: int) -> Tuple[torch.Tensor]:
    """query, p_id, gold_passage를 batch로 반환합니다."""
    batch_q = pad_sequence(
        [T(e[0]) for e in batch], batch_first=True, padding_value=padding_value
    )
    batch_q_attn_mask = (batch_q != padding_value).long()
    batch_p_id = T([e[1] for e in batch])[:, None]
    batch_p = pad_sequence(
        [T(e[2]) for e in batch], batch_first=True, padding_value=padding_value
    )
    answer = [e[3] for e in batch]
    batch_p_attn_mask = (batch_p != padding_value).long()
    return (batch_q, batch_q_attn_mask, batch_p_id, batch_p, batch_p_attn_mask, answer)


class XORQASampler(torch.utils.data.BatchSampler):
    """in-batch negative학습을 위해 batch 내에 중복 answer를 갖지 않도록 batch를 구성합니다.
    sample 일부를 pass하기 때문에 전체 data 수보다 iteration을 통해 나오는 데이터 수가 몇십개 정도 적습니다."""

    def __init__(
        self,
        data_source: Sized,
        batch_size: int,
        drop_last: bool = False,
        shuffle: bool = True,
        generator=None,
    ) -> None:
        
        if shuffle:
            sampler = torch.utils.data.RandomSampler(
                data_source, replacement=False, generator=generator
            )
        else:
            sampler = torch.utils.data.SequentialSampler(data_source)
        super(XORQASampler, self).__init__(
            sampler=sampler, batch_size=batch_size, drop_last=drop_last
        )

    def __iter__(self) -> Iterator[List[int]]:
        sampled_p_id = []
        sampled_idx = []
        for idx in self.sampler:
            item = self.sampler.data_source[idx]
            if item[1] in sampled_p_id:
                continue  # 만일 같은 answer passage가 이미 뽑혔다면 pass
            sampled_idx.append(idx)
            sampled_p_id.append(item[1])
            if len(sampled_idx) >= self.batch_size:
                yield sampled_idx
                sampled_p_id = []
                sampled_idx = []
        if len(sampled_idx) > 0 and not self.drop_last:
            yield sampled_idx


class XORQADataset:
    def __init__(self, xorqa_path: str, MODEL_TYPE = None, MODEL_NAME = None, chunk_size = 512):
        self.xorqa_path = xorqa_path
        self.data_tuples = []
        self.MODEL_NAME = MODEL_NAME
        self.MODEL_TYPE = MODEL_TYPE
        self.chunk_size = chunk_size
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        self.pad_token_id = self.tokenizer.pad_token_id
        self.load()

    @property
    def dataset(self) -> List[Tuple]:
        return self.tokenized_tuples

    def stat(self):
        """xorqa 데이터셋의 스탯을 출력합니다."""
        raise NotImplementedError()

    def load(self):
        """데이터 전처리가 완료되었다면 load하고 그렇지 않으면 전처리를 수행합니다."""
        self.xorqa_processed_path = (
            f"{self.xorqa_path.split('.json')[0]}_{self.MODEL_TYPE}_processed_{self.chunk_size}.p"
        )

        if os.path.exists(self.xorqa_processed_path):
            with open(self.xorqa_processed_path, "rb") as f:
                self.tokenized_tuples = pickle.load(f)

        else:
            self._load_data()
            self._match_passage()
            # tokenizing raw dataset
            if self.MODEL_TYPE == 'e5':
                self.tokenized_tuples = [
                    (self.tokenizer.encode('query: '+ q), d, self.tokenizer.encode(p), ans)
                    for q, d, p, ans in tqdm(self.data_tuples, desc="tokenize")
                ]
            
            else :
                self.tokenized_tuples = [
                    (self.tokenizer.encode(q), d, self.tokenizer.encode(p), ans)
                    for q, d, p, ans in tqdm(self.data_tuples, desc="tokenize")
                ]
                
            self._save_processed_dataset()

    def _load_data(self):
        with open(self.xorqa_path, "rt", encoding="utf8") as f:
            data = json.load(f)
        
        self.raw_json = data["data"]
        import pandas as pd

        open_path = f'./{self.MODEL_TYPE}_title_passage_map_{self.chunk_size}.p'
        
        with open(open_path, "rb") as f:
            self.title_passage_map = pickle.load(f)
       

    def _get_cand_ids(self, title):
        """미리 구축한 en-wiki 데이터에서 해당 title에 맞는 id들을 가지고 옵니다."""
        refined_title = None
        ret = self.title_passage_map.get(title, None)
        
        if not ret:
            refined_title = re.sub(r"\(.*\)", "", title).strip()
            ret = self.title_passage_map.get(refined_title, None)
        return ret, refined_title

    def _match_passage(self):
        """미리 구축한 en-wiki 데이터와 xorqa의 answer를 매칭하여 (query, passage_id, passage)의 tuple을 구성합니다."""
        cnt1, cnt2, cnt3, cnt4 = 0, 0, 0, 0
        for item in tqdm(self.raw_json, desc="matching silver passages"):
            title = item["title"].replace("_", " ")  # _를 공백문자로 변경
            para = item["paragraphs"]
            cand_ids, refined_title = self._get_cand_ids(title)
            if refined_title is not None and cand_ids:
                pass
            if cand_ids is None:
                
                continue
            target_file_p = get_passage_file(self.MODEL_TYPE, self.chunk_size, cand_ids)
            
            if target_file_p is None:
                
                continue
            
            with open(target_file_p, "rb") as f:
                target_file = pickle.load(f)
            contexts = {cand_id: target_file[cand_id] for cand_id in cand_ids}
            
            for p in para:
                qas = p["qas"]
                for qa in qas:
                    answer = qa["answer"][0]["text"]  # 아무 정답이나 뽑습니다.
                    answer_pos = qa["answer"][0]["answer_start"]
                    answer_clue_start = max(0, answer_pos - 5)
                    answer_clue_end = min(
                        len(p["context"]), answer_pos + len(answer) + 5
                    )
                    answer_clue = p["context"][
                        answer_clue_start:answer_clue_end
                    ]  # gold passage를 찾기 위해서 +-5칸의 주변 text 활용
                    
                    if answer not in answer_clue:
                        print('포함 안됨')
                        continue
                    question = qa["question"]
                    
                    answer_p = [
                        (p_id, c, answer) for p_id, c in contexts.items() if answer_clue in c
                    ]  # answer가 단순히 들어있는 문서를 뽑는다.
                    
                    
                    self.data_tuples.extend(
                        [(question, p_id, c, answer) for p_id, c, answer in answer_p]
                    )
                    
                    
        print(f"not found: {cnt1}, {cnt2}, {cnt3}, {cnt4}")

    def _save_processed_dataset(self):
        """전처리한 데이터를 저장합니다."""
        with open(self.xorqa_processed_path, "wb") as f:
            pickle.dump(self.tokenized_tuples, f)
        

