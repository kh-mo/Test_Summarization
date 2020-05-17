'''

트리구조 : 계층적 자료구조(hierarchical data structure)
트리의 계산 복잡도 : O()
일반트리 vs 이진트리

트리에서 헷갈리기 쉬운 용어
차수(degree) : 자식노드 수
형제노드(sibling) : 동일한 부모를 가지는 노드
서브트리(subtree) : 노드 자신과 후손노드(descendant, 노드 아래로 매달린 모든 노드 집합)로 구성된 트리
레벨(level) : 루트가 1이고 아래로 내려갈수록 1씩 증가

노드 구성방법
1. 그림4-2 : k+1개 레퍼런스 작성 노드 : 노드의 차수가 k로 고정되어 있을 때
2. 그림4-3 : 왼쪽 자식-오른쪽 형제 노드 : 노드의 차수가 일정하지 않을 때 효율적
3. 그림4-10 : 키와 래퍼런스 필드 2개 : 이진트리 구성

이진트리 : 각 노드의 자식 수가 2 이하인 트리
포화이진트리(full binary tree)
완전이진트리(complete binary tree)

레벨 k에 있는 최대 노드 수 2^(k-1)

binary search tree : binary search 개념이 들어간 tree
BST의 연산은 O(h) 시간이 걸린다
BST의 높이가 가장 낮을 때 = 완전이진트리=logN
BST의 높이가 가장 높을 때 = 편향이진트리=N
랜덤하게 N개 노드를 삽입하면 평균 트리 높이는 1.39logN
'''

class Node:
    def __init__(self, item, value, left=None, right=None):
        self.item = item
        self.value = value
        self.left = left
        self.right = right

class BinaryTree:
    def __init__(self):
        self.root = None

    def height(self, root):
        if root == None:
            return 0
        return max(self.height(root.left), self.height(root.right)) + 1

t = BinaryTree()
n1 = Node(100)
n2 = Node(200)
n3 = Node(300)
n4 = Node(400)

n1.left = n2
n1.right = n3
n2.left = n4
t.root = n1
t.height(n4)