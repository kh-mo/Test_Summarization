# 돌려보는 코드
#
# import numpy as np
# from my_sklearn.ensemble import IsolationForest
# clf = IsolationForest(n_estimators=100)
# clf.fit(np.random.randn(10, 30))
# clf.estimators_
# clf.estimators_features_
# clf._max_features
# clf._compute_score_samples(np.random.randn(10, 30),False)
#
########################################################################################################################
# my iForest
'''
PEP8 에서 _로 네이밍한 것은 private를 의미한다
즉, 해당 모듈에서만 쓰는 것을 의미한다
from module import * 할 때는 무시하는 변수, 함수들이다(다만 직접 호출하면 사용가능하다)

return self
    : 객체 자체를 반환한다, 이 명령어가 있고 없고에 따라 clf.fit().fit()과 같은 구조를 작성하는 것을 용인하거나 못하게 된다
        명령어가 존재할 경우 clf.fit().fit() 형태 사용이 가능하다
        만약 해당 명령어가 없다면 clf.fit()이 NoneType을 반환하므로 연속해서 함수를 쓸 수 없다(AttributeError 발생)
        이런 chain form이 코드를 작성하는데 가독성에 얼마나 영향을 주는지가 관건이다
        함수의 기능 자체가 실행되는 것은 chain형태를 쓰는 것과 clf.fit()을 두번 쓰는 것이 동일하다

      ex)
        print(clf.fit().fit())

        # instaed of
        clf.fit()
        clf.fit()
        print(clf)
'''
import numbers
import numpy as np
from warnings import warn

# from ..utils import check_array, check_random_state
# from ..tree import ExtraTreeRegressor

from sklearn.utils import check_array, check_random_state
from sklearn.tree import ExtraTreeRegressor

class IsolationForest:
    '''
    알고리즘 설명

    n_estimators : 트리 숫자
    max_samples : 샘플링할 데이터 수로 max_samples 파라미터가 가진 값에 따라 다른 결과가 리턴됨
        1) "auto" : 256 이하 샘플링 숫자가 자동 리턴
        2) int : 해당 값으로 숫자 리턴, 데이터 최대 샘플 수를 넘을수는 없다
        3) float : 비율을 의미, 데이터 최대 샘플 수의 일정 비율만큼 리턴
    contamination : score의 threshold값을 0으로 맞추기 위한 인자
    random_state : fit 시, 임의의 y 값 생성

    fit 함수를 실행시켰을 때 self.estimators_, self.estimators_features_, self._max_features 값을 얻을 수 있다
    '''
    def __init__(self,
                 n_estimators=100,
                 max_samples="auto",
                 contamination="auto",
                 random_state=None):
        super().__init__(
            base_estimator=ExtraTreeRegressor(max_features=1,
                                              splitter='random',
                                              random_state=random_state),
            n_estimators=n_estimators,
            max_samples=max_samples,
            random_state=random_state)

        self.contamination = contamination

    def _parallel_args(self): # private method
        return

    def fit(self, X, y=None, sample_weight=None):
        '''
        y : 사용하지 않지만 sklearn convention 유지를 위해 작성
        '''
        X = check_array(X)
        # X = check_array(X, accept_sparse=['csc'])

        rnd = check_random_state(self.random_state)
        y = rnd.uniform(size=X.shape[0])

        # max_sample은 1보다 크고 n_samples보다 작은 [1, n_samples] 범위에 있어야 한다
        n_samples = X.shape[0]

        # max_samples를 auto나, string으로 지정하면 최대 256
        if isinstance(self.max_samples, str):
            if self.max_samples == "auto":
                max_samples = min(256, n_samples)
            else:
                raise ValueError('max_samples (%s) is not supported.'
                                 'Valid choices are: "auto", int or'
                                 'float' % self.max_samples)

        # max_samples를 int로 지정하면 해당 값으로 지정, 단 그 값은 데이터 총 샘플 수를 넘을 수 없다
        elif isinstance(self.max_samples, numbers.Integral):
            if self.max_samples > n_samples:
                warn("max_samples (%s) is greater than the "
                     "total number of samples (%s). max_samples "
                     "will be set to n_samples for estimation."
                     % (self.max_samples, n_samples))
                max_samples = n_samples
            else:
                max_samples = self.max_samples

        # max_samples를 float로 지정하면 비율로 간주함, 데이터 총 샘플 수의 일정 비율만큼 구한다
        else:
            if not 0. < self.max_samples <= 1.:
                raise ValueError("max_samples must be in (0, 1], got %r"
                                 % self.max_samples)
            max_samples = int(self.max_samples * X.shape[0])

        self.max_samples_ = max_samples
        # 평균 트리 높이 : 최소 1, 최대 log2(max_sample)
        max_depth = int(np.ceil(np.log2(max(max_samples, 2))))
        super()._fit(X, y, max_samples, max=max_depth, sample_weight=sample_weight)

        if self.contamination == "auto":
            self.offset_ = -0.5
            return self
        elif isinstance(self.contamination, numbers.Integral) or isinstance(self.contamination, float):
            self.offset_ = np.percentile(self.score_samples(X), 100. * self.contamination)
            return self
        else:
            raise ValueError("해당 값은 처리 불가")

    def predict(self, X):
        return

    def decision_function(self, X):
        '''
        threshold를 0으로 만들기 위해서 offset_을 뺀다
        '''
        return self.score_samples(X) - self.offset_

    def score_samples(self, X):
        '''
        original paper에서 정의된 anomaly score와 반대
        1 : 정상
        0 : 비정상
        '''
        return -self._compute_chunked_score_samples(X)

    def _compute_chunked_score_samples(self, X): # private method

        # n_samples = _num_samples(X)
        n_samples = X.shape[0]

        if self._max_features == X.shape[1]:
            subsample_features = False
        else:
            subsample_features = True

        scores = np.zeros(n_samples)
        scores = self._compute_score_samples(X, subsample_features)

        return scores

    def _compute_score_samples(self, X, subsample_feutres): # private method
        n_samples = X.shape[0] # 들어온 sample의 갯수
        depths = np.zeros(n_samples)
        for tree, features in zip(self.estimators_, self.estimators_features_):
            # feature를 subsample 한다면 self.estimators_features_에서 얻어진 feature 집합으로 구성
            # feature를 선택하는 방식이 random인지 어떤 규칙이 있는지는 self.estimators_features_의 생성 방식을 참고해야 함
            X_subset = X[:, features] if subsample_feutres else X

            # leaf node index 반환
            leaves_index = tree.apply(X_subset)
            # (data sample by total node) matrix 생성, 데이터 샘플이 지나간 path에는 1이 할당
            node_indicator = tree.decision_path(X_subset)
            # 각 노드에 들어있는 train sample의 수 반환
            n_samples_leaf = tree.tree_.n_node_samples[leaves_index]

            # paper algorithm 3
            depths += (
                np.ravel(node_indicator.sum(axis=1))
                + _average_path_length(n_samples_leaf) # tree height limit으로 unbuilt된 subtree adjustment
                - 1.0 # root node는 경로 계산에서 제외
            )

        e_h = -depths / len(self.estimators_)
        scores = 2 ** (e_h / _average_path_length([self.max_samples_]))
        return scores

def _average_path_length(n_samples_leaf):

    n_samples_leaf = check_array(n_samples_leaf, ensure_2d=False)

    n_samples_leaf_shape = n_samples_leaf.shape
    n_samples_leaf = n_samples_leaf.reshape((1,-1))
    average_path_length = np.zeros(n_samples_leaf.shape)

    mask_1 = n_samples_leaf <= 1
    mask_2 = n_samples_leaf == 2
    not_mask = ~np.logical_or(mask_1, mask_2)

    average_path_length[mask_1] = 0.
    average_path_length[mask_2] = 1.
    average_path_length[not_mask] = (
        2.0 * (np.log(n_samples_leaf[not_mask] - 1.0) + np.euler_gamma)
        - 2.0 * (n_samples_leaf[not_mask] - 1.0) / n_samples_leaf[not_mask]
    )

    return average_path_length.reshape(n_samples_leaf_shape)



########################################################################################################################
# 참고 my_sklearn iForest
#
# import numbers
# import numpy as np
# from scipy.sparse import issparse
# from warnings import warn
#
# from ..tree import ExtraTreeRegressor
# from ..utils import (
#     check_random_state,
#     check_array,
#     gen_batches,
#     get_chunk_n_rows,
# )
# from ..utils.fixes import _joblib_parallel_args
# from ..utils.validation import check_is_fitted, _num_samples
# from ..base import OutlierMixin
#
# from ._bagging import BaseBagging
#
# __all__ = ["IsolationForest"]
#
#
# class IsolationForest(OutlierMixin, BaseBagging):
#     """
#     Isolation Forest Algorithm.
#
#     Return the anomaly score of each sample using the IsolationForest algorithm
#
#     The IsolationForest 'isolates' observations by randomly selecting a feature
#     and then randomly selecting a split value between the maximum and minimum
#     values of the selected feature.
#
#     Since recursive partitioning can be represented by a tree structure, the
#     number of splittings required to isolate a sample is equivalent to the path
#     length from the root node to the terminating node.
#
#     This path length, averaged over a forest of such random trees, is a
#     measure of normality and our decision function.
#
#     Random partitioning produces noticeably shorter paths for anomalies.
#     Hence, when a forest of random trees collectively produce shorter path
#     lengths for particular samples, they are highly likely to be anomalies.
#
#     Read more in the :ref:`User Guide <isolation_forest>`.
#
#     .. versionadded:: 0.18
#
#     Parameters
#     ----------
#     n_estimators : int, optional (default=100)
#         The number of base estimators in the ensemble.
#
#     max_samples : int or float, optional (default="auto")
#         The number of samples to draw from X to train each base estimator.
#             - If int, then draw `max_samples` samples.
#             - If float, then draw `max_samples * X.shape[0]` samples.
#             - If "auto", then `max_samples=min(256, n_samples)`.
#
#         If max_samples is larger than the number of samples provided,
#         all samples will be used for all trees (no sampling).
#
#     contamination : 'auto' or float, optional (default='auto')
#         The amount of contamination of the data set, i.e. the proportion
#         of outliers in the data set. Used when fitting to define the threshold
#         on the scores of the samples.
#
#             - If 'auto', the threshold is determined as in the
#               original paper.
#             - If float, the contamination should be in the range [0, 0.5].
#
#         .. versionchanged:: 0.22
#            The default value of ``contamination`` changed from 0.1
#            to ``'auto'``.
#
#     max_features : int or float, optional (default=1.0)
#         The number of features to draw from X to train each base estimator.
#
#             - If int, then draw `max_features` features.
#             - If float, then draw `max_features * X.shape[1]` features.
#
#     bootstrap : bool, optional (default=False)
#         If True, individual trees are fit on random subsets of the training
#         data sampled with replacement. If False, sampling without replacement
#         is performed.
#
#     n_jobs : int or None, optional (default=None)
#         The number of jobs to run in parallel for both :meth:`fit` and
#         :meth:`predict`. ``None`` means 1 unless in a
#         :obj:`joblib.parallel_backend` context. ``-1`` means using all
#         processors. See :term:`Glossary <n_jobs>` for more details.
#
#     behaviour : str, default='deprecated'
#         This parameter has not effect, is deprecated, and will be removed.
#
#         .. versionadded:: 0.20
#            ``behaviour`` is added in 0.20 for back-compatibility purpose.
#
#         .. deprecated:: 0.20
#            ``behaviour='old'`` is deprecated in 0.20 and will not be possible
#            in 0.22.
#
#         .. deprecated:: 0.22
#            ``behaviour`` parameter is deprecated in 0.22 and removed in
#            0.24.
#
#     random_state : int, RandomState instance or None, optional (default=None)
#         If int, random_state is the seed used by the random number generator;
#         If RandomState instance, random_state is the random number generator;
#         If None, the random number generator is the RandomState instance used
#         by `np.random`.
#
#     verbose : int, optional (default=0)
#         Controls the verbosity of the tree building process.
#
#     warm_start : bool, optional (default=False)
#         When set to ``True``, reuse the solution of the previous call to fit
#         and add more estimators to the ensemble, otherwise, just fit a whole
#         new forest. See :term:`the Glossary <warm_start>`.
#
#         .. versionadded:: 0.21
#
#     Attributes
#     ----------
#     estimators_ : list of DecisionTreeClassifier
#         The collection of fitted sub-estimators.
#
#     estimators_samples_ : list of arrays
#         The subset of drawn samples (i.e., the in-bag samples) for each base
#         estimator.
#
#     max_samples_ : integer
#         The actual number of samples
#
#     offset_ : float
#         Offset used to define the decision function from the raw scores. We
#         have the relation: ``decision_function = score_samples - offset_``.
#         ``offset_`` is defined as follows. When the contamination parameter is
#         set to "auto", the offset is equal to -0.5 as the scores of inliers are
#         close to 0 and the scores of outliers are close to -1. When a
#         contamination parameter different than "auto" is provided, the offset
#         is defined in such a way we obtain the expected number of outliers
#         (samples with decision function < 0) in training.
#
#     Notes
#     -----
#     The implementation is based on an ensemble of ExtraTreeRegressor. The
#     maximum depth of each tree is set to ``ceil(log_2(n))`` where
#     :math:`n` is the number of samples used to build the tree
#     (see (Liu et al., 2008) for more details).
#
#     References
#     ----------
#     .. [1] Liu, Fei Tony, Ting, Kai Ming and Zhou, Zhi-Hua. "Isolation forest."
#            Data Mining, 2008. ICDM'08. Eighth IEEE International Conference on.
#     .. [2] Liu, Fei Tony, Ting, Kai Ming and Zhou, Zhi-Hua. "Isolation-based
#            anomaly detection." ACM Transactions on Knowledge Discovery from
#            Data (TKDD) 6.1 (2012): 3.
#
#     See Also
#     ----------
#     my_sklearn.covariance.EllipticEnvelope : An object for detecting outliers in a
#         Gaussian distributed dataset.
#     my_sklearn.svm.OneClassSVM : Unsupervised Outlier Detection.
#         Estimate the support of a high-dimensional distribution.
#         The implementation is based on libsvm.
#     my_sklearn.neighbors.LocalOutlierFactor : Unsupervised Outlier Detection
#         using Local Outlier Factor (LOF).
#
#     Examples
#     --------
#     >>> from my_sklearn.ensemble import IsolationForest
#     >>> X = [[-1.1], [0.3], [0.5], [100]]
#     >>> clf = IsolationForest(random_state=0).fit(X)
#     >>> clf.predict([[0.1], [0], [90]])
#     array([ 1,  1, -1])
#     """
#
#     def __init__(self,
#                  n_estimators=100,
#                  max_samples="auto",
#                  contamination="auto",
#                  max_features=1.,
#                  bootstrap=False,
#                  n_jobs=None,
#                  behaviour='deprecated',
#                  random_state=None,
#                  verbose=0,
#                  warm_start=False):
#         super().__init__(
#             base_estimator=ExtraTreeRegressor(
#                 max_features=1,
#                 splitter='random',
#                 random_state=random_state),
#             # here above max_features has no links with self.max_features
#             bootstrap=bootstrap,
#             bootstrap_features=False,
#             n_estimators=n_estimators,
#             max_samples=max_samples,
#             max_features=max_features,
#             warm_start=warm_start,
#             n_jobs=n_jobs,
#             random_state=random_state,
#             verbose=verbose)
#
#         self.behaviour = behaviour
#         self.contamination = contamination
#
#     def _set_oob_score(self, X, y):
#         raise NotImplementedError("OOB score not supported by iforest")
#
#     def _parallel_args(self):
#         # ExtraTreeRegressor releases the GIL, so it's more efficient to use
#         # a thread-based backend rather than a process-based backend so as
#         # to avoid suffering from communication overhead and extra memory
#         # copies.
#         return _joblib_parallel_args(prefer='threads')
#
#     def fit(self, X, y=None, sample_weight=None):
#         """
#         Fit estimator.
#
#         Parameters
#         ----------
#         X : array-like or sparse matrix, shape (n_samples, n_features)
#             The input samples. Use ``dtype=np.float32`` for maximum
#             efficiency. Sparse matrices are also supported, use sparse
#             ``csc_matrix`` for maximum efficiency.
#
#         y : Ignored
#             Not used, present for API consistency by convention.
#
#         sample_weight : array-like of shape (n_samples,), default=None
#             Sample weights. If None, then samples are equally weighted.
#
#         Returns
#         -------
#         self : object
#             Fitted estimator.
#         """
#         if self.behaviour != 'deprecated':
#             if self.behaviour == 'new':
#                 warn(
#                     "'behaviour' is deprecated in 0.22 and will be removed "
#                     "in 0.24. You should not pass or set this parameter.",
#                     FutureWarning
#                 )
#             else:
#                 raise NotImplementedError(
#                     "The old behaviour of IsolationForest is not implemented "
#                     "anymore. Remove the 'behaviour' parameter."
#                 )
#
#         X = check_array(X, accept_sparse=['csc'])
#         if issparse(X):
#             # Pre-sort indices to avoid that each individual tree of the
#             # ensemble sorts the indices.
#             X.sort_indices()
#
#         rnd = check_random_state(self.random_state)
#         y = rnd.uniform(size=X.shape[0])
#
#         # ensure that max_sample is in [1, n_samples]:
#         n_samples = X.shape[0]
#
#         if isinstance(self.max_samples, str):
#             if self.max_samples == 'auto':
#                 max_samples = min(256, n_samples)
#             else:
#                 raise ValueError('max_samples (%s) is not supported.'
#                                  'Valid choices are: "auto", int or'
#                                  'float' % self.max_samples)
#
#         elif isinstance(self.max_samples, numbers.Integral):
#             if self.max_samples > n_samples:
#                 warn("max_samples (%s) is greater than the "
#                      "total number of samples (%s). max_samples "
#                      "will be set to n_samples for estimation."
#                      % (self.max_samples, n_samples))
#                 max_samples = n_samples
#             else:
#                 max_samples = self.max_samples
#         else:  # float
#             if not 0. < self.max_samples <= 1.:
#                 raise ValueError("max_samples must be in (0, 1], got %r"
#                                  % self.max_samples)
#             max_samples = int(self.max_samples * X.shape[0])
#
#         self.max_samples_ = max_samples
#         max_depth = int(np.ceil(np.log2(max(max_samples, 2))))
#         super()._fit(X, y, max_samples,
#                      max_depth=max_depth,
#                      sample_weight=sample_weight)
#
#         if self.contamination == "auto":
#             # 0.5 plays a special role as described in the original paper.
#             # we take the opposite as we consider the opposite of their score.
#             self.offset_ = -0.5
#             return self
#
#         # else, define offset_ wrt contamination parameter
#         self.offset_ = np.percentile(self.score_samples(X),
#                                      100. * self.contamination)
#
#         return self
#
#     def predict(self, X):
#         """
#         Predict if a particular sample is an outlier or not.
#
#         Parameters
#         ----------
#         X : array-like or sparse matrix, shape (n_samples, n_features)
#             The input samples. Internally, it will be converted to
#             ``dtype=np.float32`` and if a sparse matrix is provided
#             to a sparse ``csr_matrix``.
#
#         Returns
#         -------
#         is_inlier : array, shape (n_samples,)
#             For each observation, tells whether or not (+1 or -1) it should
#             be considered as an inlier according to the fitted model.
#         """
#         check_is_fitted(self)
#         X = check_array(X, accept_sparse='csr')
#         is_inlier = np.ones(X.shape[0], dtype=int)
#         is_inlier[self.decision_function(X) < 0] = -1
#         return is_inlier
#
#     def decision_function(self, X):
#         """
#         Average anomaly score of X of the base classifiers.
#
#         The anomaly score of an input sample is computed as
#         the mean anomaly score of the trees in the forest.
#
#         The measure of normality of an observation given a tree is the depth
#         of the leaf containing this observation, which is equivalent to
#         the number of splittings required to isolate this point. In case of
#         several observations n_left in the leaf, the average path length of
#         a n_left samples isolation tree is added.
#
#         Parameters
#         ----------
#         X : array-like or sparse matrix, shape (n_samples, n_features)
#             The input samples. Internally, it will be converted to
#             ``dtype=np.float32`` and if a sparse matrix is provided
#             to a sparse ``csr_matrix``.
#
#         Returns
#         -------
#         scores : array, shape (n_samples,)
#             The anomaly score of the input samples.
#             The lower, the more abnormal. Negative scores represent outliers,
#             positive scores represent inliers.
#         """
#         # We subtract self.offset_ to make 0 be the threshold value for being
#         # an outlier:
#
#         return self.score_samples(X) - self.offset_
#
#     def score_samples(self, X):
#         """
#         Opposite of the anomaly score defined in the original paper.
#
#         The anomaly score of an input sample is computed as
#         the mean anomaly score of the trees in the forest.
#
#         The measure of normality of an observation given a tree is the depth
#         of the leaf containing this observation, which is equivalent to
#         the number of splittings required to isolate this point. In case of
#         several observations n_left in the leaf, the average path length of
#         a n_left samples isolation tree is added.
#
#         Parameters
#         ----------
#         X : array-like or sparse matrix, shape (n_samples, n_features)
#             The input samples.
#
#         Returns
#         -------
#         scores : array, shape (n_samples,)
#             The anomaly score of the input samples.
#             The lower, the more abnormal.
#         """
#         # code structure from ForestClassifier/predict_proba
#         check_is_fitted(self)
#
#         # Check data
#         X = check_array(X, accept_sparse='csr')
#         if self.n_features_ != X.shape[1]:
#             raise ValueError("Number of features of the model must "
#                              "match the input. Model n_features is {0} and "
#                              "input n_features is {1}."
#                              "".format(self.n_features_, X.shape[1]))
#
#         # Take the opposite of the scores as bigger is better (here less
#         # abnormal)
#         return -self._compute_chunked_score_samples(X)
#
#     def _compute_chunked_score_samples(self, X):
#
#         n_samples = _num_samples(X)
#
#         if self._max_features == X.shape[1]:
#             subsample_features = False
#         else:
#             subsample_features = True
#
#         # We get as many rows as possible within our working_memory budget
#         # (defined by my_sklearn.get_config()['working_memory']) to store
#         # self._max_features in each row during computation.
#         #
#         # Note:
#         #  - this will get at least 1 row, even if 1 row of score will
#         #    exceed working_memory.
#         #  - this does only account for temporary memory usage while loading
#         #    the data needed to compute the scores -- the returned scores
#         #    themselves are 1D.
#
#         chunk_n_rows = get_chunk_n_rows(row_bytes=16 * self._max_features,
#                                         max_n_rows=n_samples)
#         slices = gen_batches(n_samples, chunk_n_rows)
#
#         scores = np.zeros(n_samples, order="f")
#
#         for sl in slices:
#             # compute score on the slices of test samples:
#             scores[sl] = self._compute_score_samples(X[sl], subsample_features)
#
#         return scores
#
#     def _compute_score_samples(self, X, subsample_features):
#         """
#         Compute the score of each samples in X going through the extra trees.
#
#         Parameters
#         ----------
#         X : array-like or sparse matrix
#
#         subsample_features : bool,
#             whether features should be subsampled
#         """
#         n_samples = X.shape[0]
#
#         depths = np.zeros(n_samples, order="f")
#
#         for tree, features in zip(self.estimators_, self.estimators_features_):
#             X_subset = X[:, features] if subsample_features else X
#
#             leaves_index = tree.apply(X_subset)
#             node_indicator = tree.decision_path(X_subset)
#             n_samples_leaf = tree.tree_.n_node_samples[leaves_index]
#
#             depths += (
#                 np.ravel(node_indicator.sum(axis=1))
#                 + _average_path_length(n_samples_leaf)
#                 - 1.0
#             )
#
#         scores = 2 ** (
#             -depths
#             / (len(self.estimators_)
#                * _average_path_length([self.max_samples_]))
#         )
#         return scores
#
#
# def _average_path_length(n_samples_leaf):
#     """
#     The average path length in a n_samples iTree, which is equal to
#     the average path length of an unsuccessful BST search since the
#     latter has the same structure as an isolation tree.
#     Parameters
#     ----------
#     n_samples_leaf : array-like, shape (n_samples,).
#         The number of training samples in each test sample leaf, for
#         each estimators.
#
#     Returns
#     -------
#     average_path_length : array, same shape as n_samples_leaf
#     """
#
#     n_samples_leaf = check_array(n_samples_leaf, ensure_2d=False)
#
#     n_samples_leaf_shape = n_samples_leaf.shape
#     n_samples_leaf = n_samples_leaf.reshape((1, -1))
#     average_path_length = np.zeros(n_samples_leaf.shape)
#
#     mask_1 = n_samples_leaf <= 1
#     mask_2 = n_samples_leaf == 2
#     not_mask = ~np.logical_or(mask_1, mask_2)
#
#     average_path_length[mask_1] = 0.
#     average_path_length[mask_2] = 1.
#     average_path_length[not_mask] = (
#         2.0 * (np.log(n_samples_leaf[not_mask] - 1.0) + np.euler_gamma)
#         - 2.0 * (n_samples_leaf[not_mask] - 1.0) / n_samples_leaf[not_mask]
#     )
#
#     return average_path_length.reshape(n_samples_leaf_shape)
#
#
