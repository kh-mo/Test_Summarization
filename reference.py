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
    b_index, b_value, b_score, b_lnode, b_rnode = 999, 999, 999, None, None
    for idx in range(data.shape[1] - 1):
        for row in data.index:
            threshold_value = data.iloc[row][idx]
            lnode, rnode = make_subnode(idx, threshold_value, data)
            gini_score = get_gini_score(lnode, classes) * (len(rnode) / ndata) \
                         + get_gini_score(rnode, classes) * (len(rnode) / ndata)
            if gini_score < b_score:
                b_index, b_value, b_score, b_lnode, b_rnode = idx, threshold_value, gini_score, lnode, rnode
                print('X%d < %.3f Gini=%.3f' % ((idx + 1), threshold_value, gini_score))
    return {'index': b_index, 'value': b_value, 'lnode': b_lnode, 'rnode': b_rnode}


def to_terminal(data):
    '''
    노드의 클래스가 어떤것인지 결정하는 function
    '''
    return max(set(data), key=data.count)

def split(node, max_depth, min_size, depth):
    '''
    트리 분할하기
    '''
    left, right = node['lnode'], node['rnode']
    del(node['lnode'])
    del(node['rnode'])

    if not left or not right:
        node['left'] = node['right'] = to_terminal(left+right)
        return

    if depth >= max_depth:
        node['left'], node['right'] = to_terminal(left), to_terminal(right)
        return

    if len(left) <= min_size:
        node['left'] = to_terminal(left)
    else:
        node['left'] = get_split(left)
        split(node['left'], max_depth, min_size, depth+1)

    if len(right) <= min_size:
        node['right'] = to_terminal(right)
    else:
        node['right'] = get_split(right)
        split(node['right'], max_depth, min_size, depth+1)

def build_tree(data, max_depth, min_size):
    '''
    tree 만들기
    '''
    root = get_split(data)
    split(root, max_depth, min_size, 1)
    return root


# Make a prediction with a decision tree
def predict(node, row):
    if row[node['index']] < node['value']:
        if isinstance(node['left'], dict):
            return predict(node['left'], row)
        else:
            return node['left']
    else:
        if isinstance(node['right'], dict):
            return predict(node['right'], row)
        else:
            return node['right']

if __name__ == "__main__":
    # dataset download
    name = "banknote Authentication"
    file_path = download_data(name)
    dataset = pd.read_csv(file_path, engine='c', header=None)

    import numpy as np
    from sklearn.ensemble import IsolationForest
    from sklearn import metrics

    idx = np.random.permutation(dataset.shape[0])
    train_x = dataset.loc[idx[:int(dataset.shape[0]*0.7)], :3]
    test_x = dataset.loc[idx[int(dataset.shape[0] * 0.7):], :3]
    train_y = dataset.loc[idx[:int(dataset.shape[0]*0.7)], 4]
    test_y = dataset.loc[idx[int(dataset.shape[0] * 0.7):], 4]

    clf = IsolationForest().fit(train_x)
    score = -(clf.decision_function(test_x)+clf.offset_)
    fpr, tpr, thresholds = metrics.roc_curve(test_y, score)
    metrics.auc(fpr, tpr)

    # hyperparameter
    classes = [0, 1]

    tree = build_tree(data=dataset, max_depth=1, min_size=1)
    # 부모 노드가 가진 데이터셋의 클리스 집합 s = [0,0,0,0,0,1,1,1,1,1]이 있다.
    # q 변수 p값으로 나눠 리스트 두 개, [0,0,0,0,0], [1,1,1,1,1]를 얻었다
    # 부모노드의 불순도 vs 나눠진 두 자식노드의 불순도 합
    # 작으면 분기한다
