from glob import glob
import torch
from torch import Tensor
import math
import typing
from torch import tensor as T


def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def get_wiki_filepath(data_dir):
    path = f"./{data_dir}/A*/wiki_*"
    return glob(path)


def wiki_worker_init(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    overall_start = dataset.start
    overall_end = dataset.end
    split_size = int(
        math.ceil((overall_end - overall_start) / float(worker_info.num_workers))
    )
    worker_id = worker_info.id
    # end_idx = min((worker_id+1) * split_size, len(dataset.data))
    dataset.start = overall_start + worker_id * split_size
    dataset.end = min(dataset.start + split_size, overall_end)  # index error 방지


def get_passage_file(MODEL_TYPE, chunk_size, p_id_list: typing.List[int]) -> str:
    """passage id를 받아서 해당되는 파일 이름을 반환합니다."""
    target_file = None
    p_id_max = max(p_id_list)
    p_id_min = min(p_id_list)
    glob_path =  f"./processed_passages_{MODEL_TYPE}_{chunk_size}/*.p"
    for f in glob(glob_path):
        s, e = f.split("/")[-1].split(".")[0].split("-")
        s, e = int(s), int(e)
        if p_id_min >= s and p_id_max <= e:
            target_file = f
    return target_file