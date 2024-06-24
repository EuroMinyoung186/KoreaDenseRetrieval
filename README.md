# 프로젝트명

📢 2024년 1학기 [AIKU](https://github.com/AIKU-Official) 활동으로 진행한 프로젝트입니다

## 소개

(프로젝트를 소개해주세요)

## 방법론

(문제를 정의하고 이를 해결한 방법을 가독성 있게 설명해주세요)

## 환경 설정


## 사용 방법

### Data Preparation


```
wikiextractor <xml_file_path> --no-templates
```

### Training

### Evaluation

KorQuAD and XORQA data can be downloaded from https://korquad.github.io/category/1.0_KOR.html and https://nlp.cs.washington.edu/xorqa/. Furthermore, Wiki data should be downloaded for retrieving external information. Wiki data can be downloaded from https://archive.org/download/enwiki-20190201/enwiki-20190201-pages-articles-multistream.xml.bz2 and https://archive.org/download/kowiki-20190201/kowiki-20190201-pages-articles-multistream.xml.bz2

1. We should use wikiExtractor for using wiki data
```
pip install wikiextractor
wikiextractor <xml_file_path> --no-templates
```
2. 

## 예시 결과

### Kor-Kor Retrieval Score
<img width="702" alt="그림1" src="https://github.com/EuroMinyoung186/KoreaDenseRetrieval/assets/62500006/51e50a80-304d-4240-a00e-374f50513e5e">

### Kor-Eng Retrieval Score
<img width="702" alt="그림2" src="https://github.com/EuroMinyoung186/KoreaDenseRetrieval/assets/62500006/81beee3c-859b-4947-821e-b5d59e44760d">

### Embedding Projection (t-SNE)
<img width="702" alt="그림3" src="https://github.com/EuroMinyoung186/KoreaDenseRetrieval/assets/62500006/b6ddfc4f-6ddf-4a36-9904-b1bacb7b910e">

## 팀원
| 팀원                            | 역할                                       |
| -------------------------------------- | ---------------------------------------- |
| [오원준](https://github.com/owj0421)*      | Leader, Training(Overall), Distributive processing, Paper(Abstract, Introduction, Approach, Conclusion), Bi-text dataset generation  |
| [김민영](https://github.com/EuroMinyoung186)     | Evaluation(Overall), Distributive processing, Paper(Experiments), Bi-text dataset generation |
| [박정규](https://github.com/juk1329)                          | Train(Stage2), Paper(Introduction, Experiments, Related Work) |
| [임주원](https://github.com/juooni)                           | Train(Stage1, Stage2), paper(Analtsis), MS MARCO dataset generation |

