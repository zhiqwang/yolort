# All credits go to:
# https://github.com/wiseodd/hipsternet/blob/master/hipsternet/im2col.py
# https://github.com/wiseodd/hipsternet/blob/master/hipsternet/layer.py
# Was licensed under 'no-license'

import numpy as np


def __get_im2col_indices(x_shape, field_height, field_width, stride=1):
    """
    First figure out what the size of the output should be
    """
    N, C, H, W = x_shape
    assert (H - field_height) % stride == 0
    assert (W - field_height) % stride == 0
    out_height = int((H - field_height) / stride + 1)
    out_width = int((W - field_width) / stride + 1)

    i0 = np.repeat(np.arange(field_height), field_width)
    i0 = np.tile(i0, C)
    i1 = stride * np.repeat(np.arange(out_height), out_width)
    j0 = np.tile(np.arange(field_width), field_height * C)
    j1 = stride * np.tile(np.arange(out_width), out_height)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)

    k = np.repeat(np.arange(C), field_height * field_width).reshape(-1, 1)

    return (k.astype(int), i.astype(int), j.astype(int))


def im2col(x, field_height, field_width, stride=1):
    """
    An implementation of im2col based on some fancy indexing
    """
    k, i, j = __get_im2col_indices(x.shape, field_height, field_width, stride)

    # cols: (N, C * field_height * field_width, out_height * out_width)
    cols = x[:, k, i, j]
    C = x.shape[1]
    # cols: (C * field_height * field_width, out_height * out_width)
    cols = cols.transpose(1, 2, 0).reshape(field_height * field_width * C, -1)
    
    return cols


def stride_slices(c_int, h_int, w_int, h_stride, w_stride):
    """
    Get indices of im2col result that we are interested in based
    on the provided output sizes and strides
    """
    s_indices = np.array([[[
        (w_int * h_int * c) + w_int * h + w for w in np.arange(0, w_int, w_stride)]
        for h in np.arange(0, h_int, h_stride)]
        for c in range(c_int)])

    return s_indices.reshape(-1)



