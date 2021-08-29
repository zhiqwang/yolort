# Copyright 2020 Xilinx Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Numpy Max and Avg pool implementation
"""

from .im2col import im2col, stride_slices


def max_pool(X, ksize, strides):
    return __pool(X, ksize, strides, 'Max')


def avg_pool(X, ksize, strides):
    return __pool(X, ksize, strides, 'Avg')


def __pool(X, ksize, strides, op):
    n_x, c_x, h_x, w_x = X.shape
    h_filter, w_filter = ksize
    h_stride, w_stride = strides
    
    h_out, h_int = (h_x - h_filter) / h_stride + 1, (h_x - h_filter) + 1
    w_out, w_int = (w_x - w_filter) / w_stride + 1, (w_x - w_filter) + 1

    if not h_out.is_integer() or not w_out.is_integer():
        raise Exception('Invalid output dimension!')

    h_out, w_out = int(h_out), int(w_out)

    X_col = im2col(X, h_filter, w_filter)
    stride_indices = stride_slices(c_x, h_int, w_int, h_stride, w_stride)
    #print(stride_indices)
    #print(X_col)
    X_col = X_col[stride_indices,:]
    if op == 'Max':
        X_res = X_col.max(axis=1)
    elif op == 'Mean':
        X_res = X_col.mean(axis=1)
    else:
        raise NotImplementedError(f"Currently not implemented op: {op}.")
    
    X_res = X_res.reshape(c_x, h_out, w_out, n_x)
    X_res = X_res.transpose(3, 0, 1, 2)

    return X_res
