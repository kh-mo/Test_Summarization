import numpy as np
from my_sklearn.utils.validation import check_array
from my_sklearn.ensemble._iforest import _average_path_length #IsolationForest

if __name__=="__main__":
    check_array(np.array([8]), ensure_2d=False)

    # 만족
    _average_path_length(np.array([8,21,300,400,500]))
    sa(np.array([8,21,300,400,500]))