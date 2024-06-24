from collections import defaultdict
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
import os
import pickle
import time

from glob import glob
from tqdm import tqdm


class DataChunk:
    """인풋 text를 tokenizing한 뒤에 주어진 길이로 chunking 해서 반환합니다. 이때 하나의 chunk(context, index 단위)는 하나의 article에만 속해있어야 합니다."""

    def __init__(self, MODEL_NAME, case = 0, chunk_size=100):
        self.chunk_size = chunk_size
        self.MODEL_NAME = MODEL_NAME
        self.case = case
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        
            
    def get_title_and_get_text(self, input_txt):
        titles = []
        texts = []
        for art in input_txt:
            art = art.strip()
            if not art:
                continue
            try:
                title = art.split("\n")[0].strip(">").split("title=")[1].strip('"')
                text = "".join(art.split("\n")[2:]).strip()
                titles.append(title)
                texts.append(text)
                
            except:
                print(art)
                
        return titles, texts
    
    def chunk_post_processor(self, encoded_texts, titles, texts):
        
        orig_text = []
        orig_title = []
        chunk_list = [] 
        prefix = self.tokenizer('passage : ')['input_ids'][1:-1]
        
        for idx, (title, input_ids, offset_mapping, text) in enumerate(zip(titles, encoded_texts['input_ids'], encoded_texts['offset_mapping'], texts)):
            
            
            if len(input_ids) < 5:
                continue
            if self.case == 0:
                input_ids = input_ids[1:-1]
                for start_idx in range(0, len(input_ids), self.chunk_size):
                    end_idx = min(len(input_ids), start_idx + self.chunk_size)
                    chunk = prefix + input_ids[start_idx:end_idx]
                    offset = offset_mapping[start_idx:end_idx]
                    first_right, last_right = 0, -1
                    
                    if len(offset) < 2:
                        continue

                    # [CLS]와 [SEP]는 제외하고, offset을 계산합니다. (tokenizer의 한계)
                    if offset[0][0] == offset[0][1]:
                        first_right = 1
                    if offset[-1][0] == offset[-1][1]:
                        last_right = -2
                    
                    #앞에 passage: 를 붙여줍니다.
                    orig = 'passage : ' + text[offset[first_right][0]:offset[last_right][1]]
                    
                    orig_text.append(orig)
                    orig_title.append(title)
                    chunk_list.append(chunk)
                
                
            
            else:
                for start_idx in range(0, len(input_ids), self.chunk_size):
                    end_idx = min(len(input_ids), start_idx + self.chunk_size)
                    chunk = input_ids[start_idx:end_idx]
                    offset = offset_mapping[start_idx:end_idx]
                    first_right, last_right = 0, -1
                    
                    if len(offset) < 2:
                        continue
                    if offset[0][0] == offset[0][1]:
                        first_right = 1
                    if offset[-1][0] == offset[-1][1]:
                        last_right = -2
                    
                    orig = text[offset[first_right][0]:offset[last_right][1]]
                    

                    orig_text.append(orig)
                    orig_title.append(title)
                    chunk_list.append(chunk)
            
        return  orig_text, orig_title, chunk_list
    
    def chunk(self, input_file):
        """input file format은 attardi/wikiextractor에 나온 형태를 따릅니다."""
        with open(input_file, "rt", encoding="utf8") as f:
            input_txt = f.read().strip()
        input_txt = input_txt.split(
            "</doc>"
        )  # </doc> 태그로 split하여 각 article의 제목과 본문을 parsing합니다.
        
            
            
        titles, texts = self.get_title_and_get_text(input_txt)

        try:
            encoded_txt = self.tokenizer(texts, return_offsets_mapping = True)
        except:
            print('문제 발생')
            return None, None, None
        
        return self.chunk_post_processor(encoded_txt, titles, texts)
            
            

def save_orig_passage(
    MODEL_NAME, input_path="text", passage_path="processed_passages", chunk_size=100
):
    """store original passages with unique id"""
    
    os.makedirs(passage_path, exist_ok=True)
    os.makedirs(f"{passage_path}_title", exist_ok=True)
    
    app = DataChunk(MODEL_NAME=MODEL_NAME, chunk_size=chunk_size)
    idx = 0
    
    for tmp, path in enumerate(tqdm(glob(f"./{input_path}/*/wiki_*"))):
        
        ret, title, _ = app.chunk(path)
        if ret is not None:
            to_save = {idx + i: ret[i] for i in range(len(ret))}
            to_save_title = {idx + i: title[i] for i in range(len(title))}
            with open(f"./{passage_path}/{idx}-{idx+len(ret)-1}.p", "wb") as f:
                pickle.dump(to_save, f)
            with open(f"./{passage_path}_title/{idx}-{idx+len(ret)-1}_title.p", "wb") as f:
                pickle.dump(to_save_title, f)
            idx += len(ret)
        # break


def save_title_index_map(
    index_path="title_passage_map.p", source_passage_path="processed_passages"
):
    """xorqa와 klue 데이터 전처리를 위해 title과 passage id를 맵핑합니다.
    title_index_map : dict[str, list] 형태로, 특정 title에 해당하는 passage id를 저장합니다.
    """
    
    title_files = glob(f"{source_passage_path}_title/*")
    title_id_map = defaultdict(list)
    for tf in tqdm(title_files):
        with open(tf, "rb") as _tf:
            id_passage_title_map = pickle.load(_tf)
        for (id, title) in id_passage_title_map.items():
            title_id_map[title].append(id)
    with open(index_path, "wb") as f:
        pickle.dump(title_id_map, f)
        
if __name__ == "__main__":
    # 디버깅용 main
    import argparse
    from tqdm import tqdm
    from glob import glob

    parser = argparse.ArgumentParser()
    parser.add_argument('--chunk_size', type=int, default=100)
    parser.add_argument('--model_type', type=str, default='mContriever')
    args = parser.parse_args()
    passage_path = "processed_passages" + f'_{args.model_type}'
    index_path = f'{args.model_type}_' + "title_passage_map.p"
    
    save_orig_passage(MODEL_TYPE=args.model_type, passage_path=passage_path)
    save_title_index_map(index_path=index_path, source_passage_path=passage_path)