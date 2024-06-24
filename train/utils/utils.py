import torch
from torch import Tensor
import torch.nn.functional as F
import torch.distributed as dist

def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor,
                 normalize: bool=True) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    embeddings = last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    if normalize:
        embeddings = F.normalize(embeddings, p=2, dim=1)
    return embeddings

