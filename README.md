# KoreaDenseRetriever

📢 2024년 1학기 [AIKU](https://github.com/AIKU-Official) 활동으로 진행한 프로젝트입니다

📢 2024년 1학기 AIKU Project 최우수상 수상!

## 팀원
| 팀원                            | 역할                                       |
| -------------------------------------- | ---------------------------------------- |
| [오원준](https://github.com/owj0421)*      | Leader, Training(Overall), Distributive processing, Paper(Abstract, Introduction, Approach, Conclusion), Bi-text dataset generation  |
| [김민영](https://github.com/EuroMinyoung186)     | Evaluation(Overall), Distributive processing, Paper(Experiments), Bi-text dataset generation |
| [박정규](https://github.com/juk1329)                          | Train(Stage2), Paper(Introduction, Experiments, Related Work) |
| [임주원](https://github.com/juooni)                           | Train(Stage1, Stage2), paper(Analtsis), MS MARCO dataset generation |

## 소개

**CLIR**은 사용자가 Query를 주었을 때, Query와 관련 있으면서, Query와 다른 언어로 작성된 Document를 찾는 task를 말합니다. **CLIR**은 사용자가 특정 언어에 능숙하지 않거나 특정 언어로 이루어진 결과들을 찾고 싶어하는 경우에 유용하게 사용될 수 있는 task입니다.

하지만 저희 팀은 CLIR task를 해결함에 있어 기존 방법론들에 문제가 있음을 파악하였고, 이를 해결함과 동시에 CLIR 성능을 올리고자, 해당 프로젝트를 진행하였습니다.

저희는 이번 프로젝트에서 Kor-Eng CLIR task를 해결하기 위해 pseudo-score를 활용하는 매우 간단하지만 효과적인 학습 방법론을 제안하였습니다. 

## 방법론

**CLIR task**는 2가지 Challenge를 가지고 있습니다. 첫번째는 기존의 IR task는 관련 있는 document들의 순위를 매기는 **ranking capability**뿐만 아니라, CLIR task는 ranking capability 뿐만 아니라 서로 다른 언어의 embedding을 align 시킬 수 있는 **Language translation capability** 역시 필요하다는 것입니다. 두번째는 language마다 data의 수가 **imbalance**하다는 것입니다. CLIR을 위한 데이터셋에는 상대적으로 다른 언어들보다 영어가 많은 편입니다. 그렇기에 모든 언어에 대해 일관적인 성능을 얻어내는 것은 굉장히 어렵습니다.

따라서 저희는 이러한 문제점을 해결하기 위해 **Bi-text translation dataset**을 사용하는 method를 제안하였습니다. CLIR에서 영어-한국어 pair는 매우 적지만, 고품질의 영어-한국어 pair 데이터셋은 굉장히 많습니다. 이를 활용하면, language imbalance문제와 language translation capability의 향상을 이뤄낼 수 있다고 생각하였고, 실제로 좋은 결과를 얻어냈습니다.

### Stage 1

![image](https://github.com/EuroMinyoung186/KoreaDenseRetrieval/assets/62500006/3ae07def-18f7-4790-be92-a9cb722e5153)

### Stage 2

![image](https://github.com/EuroMinyoung186/KoreaDenseRetrieval/assets/62500006/f195a528-e0cf-4423-8ddb-a5395cc0b951)



조금 더 자세하게, 저희 method는 다음 사진과 같은 2-Stage method입니다. 

Stage 1은 **Query만 번역된 MS MARCO 데이터셋**과 **KL-Divergence를** 활용하여 Base-model의 Eng-Eng retrieval 능력을 활용하여 student model의 **Eng-Kor retrieval 능력을 향상**시키는 Stage입니다. 하지만 Stage 1에서는 Query만이 한국어로 번역되기 때문에, Query와 분포가 다른 Passage는 제대로 학습되지 못하고, **Query와 Passage 간의 misalignment가 발생된다는 문제**가 생깁니다. 실제로 Embedding Projection을 통해 해당 현상을 포착하였습니다.

이를 해결하기 위해 Stage 2에는 **bi-text 데이터셋**과 **KL-Divergence**를 활용하여, **Korea query와 Korea passage간의 alignment**를 이룰 수 있도록 하여, Kor-Kor performance를 향상 시켰습니다. 이를 통해 기존 모델보다는 조금 성능이 떨어지지만 꽤 높은 Kor-Kor retrieval performance와 기존 모델과 매우 큰 차이를 보이는 Eng-Kor retrieval performance를 얻을 수 있게 되었습니다.

## 환경 설정

NVIDIA Titan XP 12GB x 8 <br/>
Colab A100 40GB x 2 <br/>
ktcloud V100 40GB x 3 <br/>

## 사용 방법


### Training

Kor-Eng bi-text translation dataset is needed for this project. You can get dataset in [AI-hub](https://www.aihub.or.kr/). Additional dataset is helpful for higher performance.

```
python train.py --multiprocessing-distributed 
```

### Evaluation

KorQuAD and XORQA data can be downloaded from [KorQuAD](https://korquad.github.io/category/1.0_KOR.html) and [XORQA](https://nlp.cs.washington.edu/xorqa/). Furthermore, Wiki data should be downloaded for retrieving external information. Wiki data can be downloaded from [enwiki](https://archive.org/download/enwiki-20190201/enwiki-20190201-pages-articles-multistream.xml.bz2) and [kowiki](https://archive.org/download/kowiki-20190201/kowiki-20190201-pages-articles-multistream.xml.bz2)

1. 먼저 Retrieval을 위한 Document를 얻어오기 위해 wiki 데이터셋을 가져옵니다.
```
pip install wikiextractor
wikiextractor <xml_file_path> --no-templates
```
2. Eng-Kor Retrieval의 경우, Retrieval을 할 수 있도록, 적절한 구조의 파일을 만듭니다.
```
python preprocess_data.py
```
3. data를 적절한 chunk_size로 나누어 줍니다.
```
python preprocess.py \
    --model_type e5 \
    --case <mean pooling : 1 / average pooling : 0> \
    --pretrained True \
    --dimension <dimension size> \
    --state_path <pth_file_path> \
    --buffer_size 50000 \ 
    --index_type <save_path> 
```
4. chunk된 data를 indexing 해줍니다.
```
accelerate launch accelerate_setting.yaml indexing.py \
    --model_type e5 \
    --world_size 8 \
    --case <mean pooling : 1 / average pooling : 0> \
    --pretrained True \
    --dimension <dimension size> \
    --state_path <pth_file_path> \
    --buffer_size 50000 \ 
    --index_type <save_path> 
```
5. retrieval.py를 활용하여 retrieval performance를 평가합니다.
```
python retrieval.py \
    --model_type <model name> \
    --cuda <cuda name>
```
## 예시 결과

### Kor-Kor Retrieval Score
<img width="702" alt="그림1" src="https://github.com/EuroMinyoung186/KoreaDenseRetrieval/assets/62500006/51e50a80-304d-4240-a00e-374f50513e5e">

### Kor-Eng Retrieval Score
<img width="702" alt="그림2" src="https://github.com/EuroMinyoung186/KoreaDenseRetrieval/assets/62500006/81beee3c-859b-4947-821e-b5d59e44760d">

### Embedding Projection (t-SNE)
<img width="702" alt="그림3" src="https://github.com/EuroMinyoung186/KoreaDenseRetrieval/assets/62500006/b6ddfc4f-6ddf-4a36-9904-b1bacb7b910e">



