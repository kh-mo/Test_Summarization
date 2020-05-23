import numbers
import numpy as np
import scipy.sparse as sp

def check_array(array, ensure_2d=True, allow_nd=False):
    '''
    입력 검증 함수

    array : 체크하거나 변환할 배열
    accept_sparse : sparse matrix 받을 수 있는 여부
    ensure_2d : 배열이 2D가 아니면 에러 발생
    allow_nd : 배열의 ndim이 2를 초과할 수 있는지 여부
    '''

    if sp.issparse(array):
        # 복소수 관련한 사항은 나중에 필요하면 digging (20.05.23)
        # sparse matrix 관련한 사항은 나중에 필요하면 digging (20.05.23)
        # _ensure_no_complex_data(array)
        # array = _ensure_sparse_format(array, accept_sparse=accept_sparse,
        #                               dtype=dtype, copy=copy,
        #                               force_all_finite=force_all_finite,
        #                               accept_large_sparse=accept_large_sparse)
        raise NotImplementedError
    else:
        if ensure_2d:
            # If input is scalar raise error
            if array.ndim == 0:
                raise ValueError(
                    "ensure_2D 옵션을 True 했는데 입력받은 array는 scalar인 경우 발생한 오류\n"
                    "Expected 2D array, got scalar array instead:\narray={}.\n"
                    "Reshape your data either using array.reshape(-1, 1) if "
                    "your data has a single feature or array.reshape(1, -1) "
                    "if it contains a single sample.".format(array))
            # If input is 1D raise error
            if array.ndim == 1:
                raise ValueError(
                    "ensure_2D 옵션을 True 했는데 입력받은 array는 1D인 경우 발생한 오류\n"
                    "Expected 2D array, got 1D array instead:\narray={}.\n"
                    "Reshape your data either using array.reshape(-1, 1) if "
                    "your data has a single feature or array.reshape(1, -1) "
                    "if it contains a single sample.".format(array))

        if not allow_nd and array.ndim >= 3:
            raise ValueError("Found array with dim %d. expected <= 2."
                             % (array.ndim))

        array = np.array(array)

    return array

def check_random_state(seed):
    '''
    seed를 np.random.RandomState 인스턴스로 만듬

    seed : seed 파라미터가 가진 값에 따라 다른 결과가 리턴됨
        1) seed = None, np.random.Random의 싱글톤(singleton)이 리턴
        2) seed = int, 해당 seed값에 대한 np.random.RandomState 인스턴스 리턴
        3) seed = np.random.RandomState, seed 그대로 리턴
    '''
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, numbers.Integral):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
                     ' instance' % seed)

# sparse matrix 관련한 사항은 나중에 필요하면 digging (20.05.23)
# def _ensure_sparse_format(spmatrix, accept_sparse, dtype, copy,
#                           force_all_finite, accept_large_sparse):
#     '''
#
#     :param spmatrix:
#     :param accept_sparse:
#     :param dtype:
#     :param copy:
#     :param force_all_finite:
#     :param accept_large_sparse:
#     :return:
#     '''
#
#     """Convert a sparse matrix to a given format.
#
#     Checks the sparse format of spmatrix and converts if necessary.
#
#     Parameters
#     ----------
#     spmatrix : scipy sparse matrix
#         Input to validate and convert.
#
#     accept_sparse : string, boolean or list/tuple of strings
#         String[s] representing allowed sparse matrix formats ('csc',
#         'csr', 'coo', 'dok', 'bsr', 'lil', 'dia'). If the input is sparse but
#         not in the allowed format, it will be converted to the first listed
#         format. True allows the input to be any format. False means
#         that a sparse matrix input will raise an error.
#
#     dtype : string, type or None
#         Data type of result. If None, the dtype of the input is preserved.
#
#     copy : boolean
#         Whether a forced copy will be triggered. If copy=False, a copy might
#         be triggered by a conversion.
#
#     force_all_finite : boolean or 'allow-nan', (default=True)
#         Whether to raise an error on np.inf, np.nan, pd.NA in X. The
#         possibilities are:
#
#         - True: Force all values of X to be finite.
#         - False: accepts np.inf, np.nan, pd.NA in X.
#         - 'allow-nan': accepts only np.nan and pd.NA values in X. Values cannot
#           be infinite.
#
#         .. versionadded:: 0.20
#            ``force_all_finite`` accepts the string ``'allow-nan'``.
#
#         .. versionchanged:: 0.23
#            Accepts `pd.NA` and converts it into `np.nan`
#
#     Returns
#     -------
#     spmatrix_converted : scipy sparse matrix.
#         Matrix that is ensured to have an allowed type.
#     """
#     if dtype is None:
#         dtype = spmatrix.dtype
#
#     changed_format = False
#
#     if isinstance(accept_sparse, str):
#         accept_sparse = [accept_sparse]
#
#     # Indices dtype validation
#     _check_large_sparse(spmatrix, accept_large_sparse)
#
#     if accept_sparse is False:
#         raise TypeError('A sparse matrix was passed, but dense '
#                         'data is required. Use X.toarray() to '
#                         'convert to a dense numpy array.')
#     elif isinstance(accept_sparse, (list, tuple)):
#         if len(accept_sparse) == 0:
#             raise ValueError("When providing 'accept_sparse' "
#                              "as a tuple or list, it must contain at "
#                              "least one string value.")
#         # ensure correct sparse format
#         if spmatrix.format not in accept_sparse:
#             # create new with correct sparse
#             spmatrix = spmatrix.asformat(accept_sparse[0])
#             changed_format = True
#     elif accept_sparse is not True:
#         # any other type
#         raise ValueError("Parameter 'accept_sparse' should be a string, "
#                          "boolean or list of strings. You provided "
#                          "'accept_sparse={}'.".format(accept_sparse))
#
#     if dtype != spmatrix.dtype:
#         # convert dtype
#         spmatrix = spmatrix.astype(dtype)
#     elif copy and not changed_format:
#         # force copy
#         spmatrix = spmatrix.copy()
#
#     if force_all_finite:
#         if not hasattr(spmatrix, "data"):
#             warnings.warn("Can't check %s sparse matrix for nan or inf."
#                           % spmatrix.format, stacklevel=2)
#         else:
#             _assert_all_finite(spmatrix.data,
#                                allow_nan=force_all_finite == 'allow-nan')
#
#     return spmatrix
