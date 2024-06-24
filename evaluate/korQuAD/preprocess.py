import argparse
import pickle
import os
import torch
import indexers
import yaml

from chunk_data import DataChunk, save_orig_passage, save_title_index_map
from utils import get_wiki_filepath
from tqdm import tqdm
from transformers import AutoTokenizer

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ['CUDA_VISIBLE_DEVICES'] = '6,7'
os.environ["ACCELERATE_CONFIG_FILE"] = "./default_config.yaml"

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
parser.add_argument('--batch_size', type=int, default=2048)
parser.add_argument('--state_path', type=str, default='model_best.pth')
parser.add_argument('--skip_chunk', type=int, default=False)
parser.add_argument('--skip_save', type=int, default=False)
parser.add_argument('--cuda', type=str, default='cuda:5')
parser.add_argument('--buffer_size', type=int, default=50000)
parser.add_argument('--skip_index', type=bool, default=False)
parser.add_argument('--chunk', type=int, default=128)

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
    passage_path = passage_path + f'_{args.model_type}' #processed_passages_e5
    index_path = f'{args.model_type}_' + index_path #e5_title_passage_map.p
    model_name = modelTypeTomodelName[args.model_type] #intfloat/multilingual-e5-small


    
     # 예시로 사용된 파라미터
    save_path = "./all_wiki_data"
    indexer_output = f"2050iter_flat_{args.model_type}" #2050iter_flat_e5
    print(args.case == 0)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    chunk_sizes = [args.chunk]
    for chunk_size in chunk_sizes:
        chunker = DataChunk(MODEL_NAME=model_name, case=0, chunk_size=chunk_size - 2)
        if not args.skip_chunk:
            chunker = DataChunk(MODEL_NAME=model_name, case=0, chunk_size=chunk_size - 2) # 그대로 / case / chunk_size
            print('Doing chunking...')
            save_orig_passage(MODEL_NAME=model_name, passage_path=passage_path + f"_{chunk_size}", chunk_size=chunk_size - 2) # 그대로 / processed_passages_e5_128.p / chunk_size 

            print('saving title information...')
            save_title_index_map(index_path=index_path+ f"_{chunk_size}", source_passage_path=passage_path + f"_{chunk_size}")
        
        if not args.skip_save :
            print('saving wiki passage...')
            wiki_stream = WikiArticleStream(get_wiki_filepath('Extractor'), chunker, save_path + f"_{chunk_size}.pkl")
            for passage in tqdm(wiki_stream):
                continue
        

