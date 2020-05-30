'''
https://machinelearningmastery.com/implement-decision-tree-algorithm-scratch-python/
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

def gini_index(data, classes):
    '''
    data : 1d list, ex) [0,0,0,1,0,1,0]
    classes : 1d list, ex) [0,1]
    '''
    score = 0.0
    data_size = len(data)
    for class_val in classes:
        p = data.count(class_val) / data_size
        score += p * p
    return score


if __name__ == "__main__":
    # dataset download
    name = "banknote Authentication"
    file_path = download_data(name)
    dataset = pd.read_csv(file_path, engine='c', header=None)


    # 부모 노드가 가진 데이터셋의 클리스 집합 s = [0,0,0,0,0,1,1,1,1,1]이 있다.
    # q 변수 p값으로 나눠 리스트 두 개, [0,0,0,0,0], [1,1,1,1,1]를 얻었다
    # 부모노드의 불순도 vs 나눠진 두 자식노드의 불순도 합
    # 작으면 분기한다

    # calculate gini index
    s = [0,0,0,0,0,1,1,1,1,1]
    cs1, cs2 = [0,0,0,0,0], [1,1,1,1,1]
    classes = [0,1]

    gini_index(s, classes)
    gini_index(cs1, classes)
    gini_index(cs2, classes)
    gini_index([0,0,0,1,0,1], classes)


def split_decision(data, q, p):
    parents_score = gini_index(s, classes)
    child_score = gini_index(cs1, classes) * (len(cs1) / len(s))\
                  + gini_index(cs2, classes) * (len(cs2) / len(s))
    if parents_score > child_score:
        print("분기 안함")
    else:
        print("분기함")





