## 같은 의미를 가진 데이터를 체크하여 nC2만큼 데이터 augmentation을 수행
## a=b, b=c 라면 a=c인 데이터셋을 추가로 만드는 작업

import os
import numpy as np
from itertools import permutations

train_data = ["휴대폰 불법복제란 무엇인가요?\t휴대폰복제에대해서궁금합니다.",
              "휴대폰 불법복제란 무엇인가요?\t휴대폰 복제란?",
              "휴대폰복제에대해서궁금합니다.\t휴대폰 복제란?",
              "휴대폰 불법복제란 무엇인가요?\t불법 복제가 뭔가요??",
              "휴대폰 불법복제란 무엇인가요?\t복제란 무엇인가?",
              "휴대폰 불법복제란 무엇인가요?\t복제란 무엇인가요 ?"]
train_label = [1,1,1,0,0,0]

q_list = []
a_list = []

for datum in train_data:
    q_list.append(datum.split('\t')[0])
    a_list.append(datum.split('\t')[1])
labels_list = np.array([[np.float32(x)] for x in train_label])

pos_list = []
for uni_quest in list(set(q_list)):
    pos_tmp = [uni_quest]
    for quest in range(len(q_list)):
        if q_list[quest] == uni_quest:
            if labels_list[quest][0] == 1.0:
                pos_tmp.append(a_list[quest])
        else:
            continue
    pos_list.append(pos_tmp)

total_pos_data = []
total_pos_label = []
for lt in pos_list:
    per = permutations(lt, 2)
    for sample in list(per):
        total_pos_data.append(sample[0] + '\t' + sample[1])
        total_pos_label.append([1.])

result = train_data + total_pos_data