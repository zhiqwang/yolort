# All credits go to:
# https://github.com/wiseodd/hipsternet/blob/master/hipsternet/im2col.py
# https://github.com/wiseodd/hipsternet/blob/master/hipsternet/layer.py
# Was licensed under 'no-license'

# Used for experimental purposes

import numpy as np

from .im2col import im2col, stride_slices


def conv2d(X, W, strides):
    n_filters, d_filter, h_filter, w_filter = W.shape
    n_x, c_x, h_x, w_x = X.shape
    h_stride, w_stride = strides
    
    h_out, h_int = (h_x - h_filter) / h_stride + 1, (h_x - h_filter) + 1
    w_out, w_int = (w_x - w_filter) / w_stride + 1, (w_x - w_filter) + 1

    if not h_out.is_integer() or not w_out.is_integer():
        raise Exception('Invalid output dimension!')

    h_out, w_out = int(h_out), int(w_out)

    X_col = im2col(X, h_filter, w_filter)
    stride_indices = stride_slices(c_x, h_int, w_int, h_stride, w_stride)

    X_col = X_col[:,stride_indices]

    W_col = W.reshape(n_filters, -1)

    out = np.matmul(W_col, X_col)
    out = out.reshape(n_filters, h_out, w_out, n_x)
    out = out.transpose(3, 0, 1, 2)

    cache = (X, W, X_col)

    return out, cache