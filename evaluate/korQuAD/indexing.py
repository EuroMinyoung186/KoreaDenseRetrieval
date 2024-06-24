import argparse
import pickle
import os
import torch
import indexers
import yaml

from utils import average_pool, mean_pooling
from dataset import WikiDatset, wiki_collator
from encoder import Encoder
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.nn import functional as F
from transformers import AutoTokenizer
from accelerate import Accelerator

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'

#model_type과 model_name을 mapping하기 위한 dictionary
modelTypeTomodelName = {'e5' : 'intfloat/multilingual-e5-small', 'miniLM' : 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'}
passage_path = "processed_passages"
index_path = "title_passage_map.p"

parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--model_type', type=str, default='e5')
parser.add_argument('--world_size', type=int, default=8)
parser.add_argument('--case', type=str, default=0)
parser.add_argument('--pretrained', type=bool, default=False)
parser.add_argument('--dimension', type=int, default=384)
parser.add_argument('--indexer_type', type=str, default='DenseFlatIndexer')
parser.add_argument('--batch_size', type=int, default=32768)
parser.add_argument('--state_path', type=str, default='model_best.pth')
parser.add_argument('--skip_chunk', type=int, default=False)
parser.add_argument('--skip_save', type=int, default=False)
parser.add_argument('--cuda', type=str, default='cuda:5')
parser.add_argument('--buffer_size', type=int, default=50000)
parser.add_argument('--skip_index', type=bool, default=False)
parser.add_argument('--chunk', type=int, default=128)
parser.add_argument('--index_type', type=str, default='naive')

class WikiArticleStream:
    def __init__(self, wiki_paths, chunker, save_path):
        """
        wiki_paths: 처리할 위키 파일 경로 리스트
        chunker: 데이터를 처리할 chunker 인스턴스
        save_path: 데이터를 저장할 경로
        """
        super(WikiArticleStream, self).__init__()
        self.chunker = chunker
        self.wiki_paths = wiki_paths
        self.path_format = save_path
        self.save_path = save_path
        self.ensure_save_path_exists()

    def ensure_save_path_exists(self):
        # 저장 경로가 있는지 확인하고, 없다면 생성
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)

    def __iter__(self):
        # 여러 파일을 처리하여 하나의 파일에 저장
        idx = 0
        path_cnt = 0
            
        with open(self.save_path, 'wb') as file:
            for path in tqdm(self.wiki_paths, total = len(self.wiki_paths)):
                
                path_cnt += 1
            
                _, _, passages = self.chunker.chunk(path)
                for passage in passages:
                    # pickle을 사용하여 파일에 저장
                    pickle.dump((idx, passage), file)
                    idx += 1
                    yield passage # 데이터를 yield
                    
def save_wiki_info(wiki_paths, chunker, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'wb') as file:
        idx = 0
        for path in tqdm(wiki_paths, total = len(wiki_paths)):
            _, _, passages = chunker.chunk(path)
            for passage in passages:
                # pickle을 사용하여 파일에 저장
                pickle.dump((idx, passage), file)
                idx += 1


def get_indexer(args):
    indexer = getattr(indexers, args.indexer_type)()
    indexer.init_index(args.dimension)
    return indexer


if __name__ == '__main__':
    args = parser.parse_args()
    passage_path = passage_path + f'_{args.model_type}'
    index_path = f'{args.model_type}_' + index_path
    model_name = modelTypeTomodelName[args.model_type]
    cut = 2000

    save_path = "./all_wiki_data"
    indexer_output = f"2050iter_flat_{args.model_type}"
    state_path = f'/home/aikusrv01/aiku/minoi_big/base_enen2small_koen.pth'

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    bmodel = Encoder(model_name, args.pretrained, state_path)
    chunk_sizes = [args.chunk]
    

    for chunk_size in chunk_sizes:
        if chunk_size != 1:
            if not args.skip_index:
                
                accelerator = Accelerator()
                model = accelerator.prepare(bmodel)
                if accelerator.is_main_process:
                    print('Get Indexer...')
                    indexer = get_indexer(args)

                accelerator.wait_for_everyone()

                print('get DataLoader...')
                dataset = WikiDatset(save_path + f"_{chunk_size}.pkl")
                dataloader = DataLoader(
                    dataset, batch_size=args.batch_size // chunk_size, collate_fn=wiki_collator, shuffle=False, num_workers=0
                ) 


                # Accelerator 설정
                dataloader = accelerator.prepare(dataloader)

                print('Indexing')
                _to_index = []
                global_cur = 0
                accelerator.wait_for_everyone()
                for batch in tqdm(dataloader, desc='indexing'):
                    
                    idx, p, p_mask = batch

                    with torch.no_grad():
                        if args.case == 0:    
                            p_emb = model(p, p_mask)
                            p_emb = average_pool(p_emb.last_hidden_state, p_mask)
                            p_emb = F.normalize(p_emb, dim=-1)
                        else:
                            print('here')
                            p_emb = model(p, p_mask)
                            p_emb = mean_pooling(p_emb, p_mask)
                            p_emb = F.normalize(p_emb, dim = -1)

                    emb_list = accelerator.gather(p_emb)
                    idx = torch.tensor(idx).to(accelerator.device)
                    idx_list = accelerator.gather(idx)

                    
                    
                    if accelerator.is_main_process:
                        batch_indices = [(int(i.cpu()), emb.cpu().numpy()) for emb, i in zip(emb_list, idx_list)]
                        
                        _to_index.extend(batch_indices)

                        if len(_to_index) >= args.buffer_size - args.batch_size:
                            _to_index = sorted(_to_index, key=lambda x: x[0])
                            indexer.index_data(_to_index)
                            _to_index = []
                    
                    accelerator.wait_for_everyone()
                    

                if accelerator.is_main_process:
                    if _to_index:
                        _to_index = sorted(_to_index, key=lambda x: x[0])
                        indexer.index_data(_to_index)
                        _to_index = []

                    indexer.serialize(indexer_output + f"_{chunk_size}" + f"_{args.index_type}")


        

    
    






