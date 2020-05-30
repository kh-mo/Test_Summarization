# Isolation Forest

scikit-learn의 isolation forest를 기반으로 제작.

## 목표
- sklearn에 의존하지 않고 tree 모델 구현하기(pytorch, tensorflow 처럼)
- paper에 적힌 데이터 셋 성능 재현하기

## 성능평가

- 직접 제작한 데이터 셋(총 3개 cluster : normal, abnormal in tr&val&ts, unseen abnormal only in ts)
- official dataset
    1. [UCI banknote authentication Data Set](http://archive.ics.uci.edu/ml/datasets/banknote+authentication)
    2.

## module

#### sklearn folder
- scikit-learn iForest 코드 따라만들어 보기

#### tree_study.py
- tree 자료구조 요약 정리

#### showDataset.ipynb

#### generateDataset.py
data 폴더 생성.
data---officialSet
     |-generatedSet