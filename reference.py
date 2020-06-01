'''
https://machinelearningmastery.com/implement-decision-tree-algorithm-scratch-python/ 참조
'''

import os
import urllib.request
import pandas as pd

def download_data(name):
    '''
    name : download 가능한 데이터 셋 이름
    '''
    urls = {"BanknoteAuthentication" :
                "http://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt"}

    # dataset 이름 format 변경
    dataset_name = ""
    for ele in name.split():
        dataset_name += ele[0].upper() + ele[1:]
    dataset_path = os.path.join(os.getcwd(), "data", dataset_name)

    # 폴더 생성
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)

    # dataset 다운로드
    url = urls[dataset_name]
    print('Downloading ' + url)
    dataset = urllib.request.urlopen(url).read()
    file_path = os.path.join(dataset_path, dataset_name+".txt")
    with open(file_path, 'wb') as f:
        f.write(dataset)

    return file_path

def get_gini_score(data, classes):
    '''
    sigma{pi * (1-pi)} = 1 - (pi)^2

    data : 1d list, ex) [0,0,0,1,0,1,0]
    classes : 1d list, ex) [0,1]
    '''
    score = 0.0
    data_size = len(data)
    # data_size가 0 = node가 비어있음 = 데이터 없음 = 매우 순수한 상태 = 0 반환
    if data_size == 0:
        return 0
    for class_val in classes:
        p = data.count(class_val) / data_size
        score += p * p
    gini_score = 1 - score
    return gini_score

def make_subnode(index, value, data):
    '''
    특정 변수가 기준값보다 작은 데이터는 왼쪽 노드, 크거나 같은 데이터는 오른쪽 노드에 할당

    index : split 기준이 될 column index
    value : split 기준 값
    data : 데이터 셋(현재는 pandas)
    '''
    lnode, rnode = [], []
    for row in data.index:
        if data.iloc[row][index] < value:
            lnode.append(data.iloc[row][4])
        else:
            rnode.append(data.iloc[row][4])
    return lnode, rnode

def get_split(data, classes):
    ndata = data.shape[0]
    b_index, b_value, b_score = 999, 999, 999
    for idx in range(data.shape[1] - 1):
        for row in data.index:
            lnode, rnode = make_subnode(idx, data.iloc[row][idx], data)
            gini_score = get_gini_score(lnode, classes) * (len(rnode) / ndata) \
                         + get_gini_score(rnode, classes) * (len(rnode) / ndata)
            if gini_score < b_score:
                b_index, b_value, b_score = idx, data.iloc[row][idx], gini_score
                print('X%d < %.3f Gini=%.3f' % ((idx + 1), data.iloc[row][idx], gini_score))
    return {'index': b_index, 'value': b_value}

if __name__ == "__main__":
    # dataset download
    name = "banknote Authentication"
    file_path = download_data(name)
    dataset = pd.read_csv(file_path, engine='c', header=None)

    # hyperparameter
    classes = [0, 1]

    get_split(dataset, classes)
    # 부모 노드가 가진 데이터셋의 클리스 집합 s = [0,0,0,0,0,1,1,1,1,1]이 있다.
    # q 변수 p값으로 나눠 리스트 두 개, [0,0,0,0,0], [1,1,1,1,1]를 얻었다
    # 부모노드의 불순도 vs 나눠진 두 자식노드의 불순도 합
    # 작으면 분기한다

    # calculate gini index
    s = [0,0,0,0,0,1,1,1,1,1]
    cs1, cs2 = [0,0,0,0,0], [1,1,1,1,1]

    gini_index(s, classes)
    gini_index(cs1, classes)
    gini_index(cs2, classes)
    gini_index([0,0,0,1,0,1], classes)

data = dataset
