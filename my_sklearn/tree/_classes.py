from abc import ABCMeta

# from ..base import BaseEstimator
# from ..base import MultiOutputMixin

from sklearn.base import BaseEstimator
from sklearn.base import MultiOutputMixin

class BaseDecisionTree(MultiOutputMixin, BaseEstimator, metaclass=ABCMeta):
    def __init__(self,
                 splitter,
                 max_features,
                 random_state):
        self.splitter = splitter
        self.max_features = max_features
        self.random_state = random_state


class DecisionTreeRegressor(BaseDecisionTree):
    def __init__(self,
                 splitter="best",
                 max_features=None,
                 random_state=None):
        super().__init__(splitter=splitter,
                         max_features=max_features,
                         random_state=random_state)

    def fit(self):
        return

class ExtraTreeRegressor(DecisionTreeRegressor):
    def __init__(self,
                 splitter='random',
                 max_features="auto",
                 random_state=None):
        super().__init__(splitter=splitter,
                         max_features=max_features,
                         random_state=random_state)
