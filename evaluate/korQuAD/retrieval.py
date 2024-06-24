import torch
from torch import tensor as T
import pickle
import argparse
import numpy as np
from collections import defaultdict
from tqdm import tqdm

from indexers import DenseFlatIndexer
from encoder import Encoder
import torch.nn.functional as F
from dpr_data import KorQuadDataset, KorQuadSampler, korquad_collator
from utils import get_passage_file, average_pool, mean_pooling
from typing import List
import bisect
from glob import glob

class KorDPRRetriever:
    def __init__(self, model, valid_dataset, index, val_batch_size: int = 512, device='cuda:5', MODEL_TYPE = 'mContriever', case = 1):
        self.model = model.to(device)
        self.device = device
        self.tokenizer = valid_dataset.tokenizer
        self.val_batch_size = val_batch_size
        self.MODEL_TYPE = MODEL_TYPE
        self.model_type = MODEL_TYPE
        self.case = case
        self.valid_loader = torch.utils.data.DataLoader(
            dataset=valid_dataset.dataset,
            batch_sampler=KorQuadSampler(
                valid_dataset.dataset, batch_size=val_batch_size, drop_last=False
            ),
            collate_fn=lambda x: korquad_collator(
                x, padding_value=valid_dataset.pad_token_id
            ),
            num_workers = 4,
        )
        self.index = index
        self.start_list, self.end_list = self.get_num()
    def get_num(self):
        glob_path =  f"./processed_passages_{self.MODEL_TYPE}_128/*.p"
        start_list, end_list = [], []
        for f in glob(glob_path):
            s, e = f.split("/")[-1].split(".")[0].split("-")
            s, e = int(s), int(e)
            start_list.append(s)
            end_list.append(e)
        start_list = sorted(start_list)
        end_list = sorted(end_list)
        return start_list, end_list

    def find_or_previous_value(self, sorted_list, target):
        index = bisect.bisect_left(sorted_list, target)
        
        if index < len(sorted_list) and sorted_list[index] == target:
            return sorted_list[index]  # target이 리스트에 있는 경우 해당 값을 반환
        elif index > 0:
            return sorted_list[index - 1]  # target이 리스트에 없는 경우 그보다 작은 값 반환
        else:
            return None
        
    def find_or_next_value(self, sorted_list, target):
        index = bisect.bisect_right(sorted_list, target)
    
        # target이 리스트에 있는 경우 해당 값을 반환
        if index > 0 and sorted_list[index - 1] == target:
            return sorted_list[index - 1]
        elif index < len(sorted_list):
            return sorted_list[index]  # target보다 큰 값 중 가장 작은 값을 반환
        else:
            return None
        
    def get_passage_file(self, chunk_size, p_id_list):
        glob_path =  f"./processed_passages_{self.MODEL_TYPE}_128/*.p"
        s = min(p_id_list)
        e = max(p_id_list)
        start = self.find_or_previous_value(self.start_list, s)
        end = self.find_or_next_value(self.end_list, e)
        if start is None or end is None:
            return None
        return f"./processed_passages_{self.MODEL_TYPE}_128/{start}-{end}.p"
    
    def val_top_k_acc(self, case, chunk_size, k = [1,5,10,20,50,100]):
        '''validation set에서 top k 정확도를 계산합니다.'''
        
        self.model.eval()  # 평가 모드
        k_max = max(k)
        sample_cnt = 0
        retr_cnt = defaultdict(int)
        neretr_cnt = defaultdict(int)
        mrr_log = defaultdict(list)
        cut = 2000
        with torch.no_grad():

            err_cnt = 0
            for batch in tqdm(self.valid_loader, desc='valid'):
                if i >= 10000:
                    break
                q, q_mask, _, a, a_mask, answer = batch
                q, q_mask = (
                    q.to(self.device),
                    q_mask.to(self.device),
                )
                
                if self.case == 0:
                    q_emb = self.model(q, q_mask)
                    q_emb = average_pool(q_emb.last_hidden_state, q_mask)
                    q_emb = F.normalize(q_emb, dim=-1)
                
                else:
                    q_emb = self.model(q, q_mask)
                    q_emb = mean_pooling(q_emb, q_mask)
                    q_emb = F.normalize(q_emb, dim=-1)
                    
                
                result = self.index.search_knn(query_vectors=q_emb.cpu().numpy(), top_docs=k_max)
                
                
                for ((pred_idx_lst, _), _a , _a_mask, ans) in tqdm(zip(result, a, a_mask, answer), total = len(result)):
                    docs = [pickle.load(open(self.get_passage_file(chunk_size=chunk_size, p_id_list=[idx]),'rb'))[idx] for idx in pred_idx_lst]
                    
                    for _k in k:
                        tmp = [doc for doc in docs[:_k]]
                        if ans in ' '.join(tmp): 
                            retr_cnt[_k] += 1
                            try:
                                for _idx, _doc in enumerate(tmp):

                                    if ans in _doc:
                                        
                                        mrr_log[_k].append(1/(1+_idx))
                                        break
                            except:
                                if _k == 5:
                                    mrr_log[_k].append(1/_k)
                                    err_cnt += 1
                        else:
                            neretr_cnt[_k] += 1
                            mrr_log[_k].append(0)

                bsz = q.size(0)
                sample_cnt += bsz
        retr_acc = defaultdict(float)
        retr_mrr = defaultdict(float) 

        for _k in k:
            retr_acc[_k] = float(retr_cnt[_k]) / float(sample_cnt)
            retr_mrr[_k] = np.mean(mrr_log[_k])
        
        with open(f'{self.model_type}_retr_acc_result_{chunk_size}.txt', 'w') as f:
            f.write(str(retr_acc))
        with open(f'{self.model_type}_retr_mrr_result_{chunk_size}.txt', 'w') as f:
            f.write(str(retr_mrr))    
        
        return retr_acc, retr_mrr


    def retrieve(self, query: str, k: int = 100):
        """주어진 쿼리에 대해 가장 유사도가 높은 passage를 반환합니다."""
        self.model.eval()  # 평가 모드[]
        tok = self.tokenizer.batch_encode_plus([query])
        with torch.no_grad():
            out = self.model(T(tok["input_ids"]), T(tok["attention_mask"]), "query")
        result = self.index.search_knn(query_vectors=out.cpu().numpy(), top_docs=k)

        # 원문 가져오기
        passages = []
        for idx, sim in zip(*result[0]):
            path = get_passage_file([idx])
            if not path:
                print(f"No single passage path for {idx}")
                continue
            with open(path, "rb") as f:
                passage_dict = pickle.load(f)
            print(f"passage : {passage_dict[idx]}, sim : {sim}")
            passages.append((passage_dict[idx], sim))
        return passages


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default='mContriever')
    parser.add_argument("--cuda", type=str, default='cuda:7')
    args = parser.parse_args()
    checkpoint = torch.load('/home/aikusrv01/aiku/fuck/checkpoints/score/model_best.pth', map_location='cpu')
    new_state_dict = {key.replace('module.', ''): value for key, value in checkpoint['state_dict'].items()}
    model = Encoder(MODEL_TYPE=args.model_type, new_state_dict=new_state_dict, rank=args.cuda)
    model.eval()
    
    
    
    valid_dataset = KorQuadDataset("/home/aikusrv01/aiku/kortrieval/tmp_eval/korQuAD/KorQuAD_v1.0_dev.json", MODEL_TYPE=args.model_type)
    
    print('탈출')
    index = DenseFlatIndexer()
    index.deserialize(path=f"./2050iter_flat_e5")
    retriever = KorDPRRetriever(model=model, valid_dataset=valid_dataset, index=index, device=args.cuda, MODEL_TYPE=args.model_type)
    retr_acc, retr_mrr = retriever.val_top_k_acc()
    print(f"retr_acc : {retr_acc} \n 'retr_mrr : {retr_mrr}")

