# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""External function interface to CuDNN v7 library."""
# pylint: disable-msg=C0103
import ctypes
import numpy as np
from .. import api as _api
from .. import intrin as _intrin
from .. import get_global_func as _get_global_func

def _prepare_global_func_params(dims,
                                pad,
                                stride,
                                dilation,
                                x_shape=None,
                                w_shape=None):
    full_dims = dims + 2
    if x_shape:
        assert isinstance(x_shape, list)
        assert len(x_shape) == full_dims
    if w_shape:
        assert isinstance(w_shape, list)
        assert len(w_shape) == full_dims

    pad = np.full(dims, pad, dtype=np.int32) if isinstance(pad, int) \
        else np.array(pad, dtype=np.int32)
    stride = np.full(dims, stride, dtype=np.int32) if isinstance(stride, int) \
        else np.array(stride, dtype=np.int32)
    dilation = np.full(dims, dilation, dtype=np.int32) if isinstance(dilation, int) \
        else np.array(dilation, dtype=np.int32)

    xshape = np.array(x_shape, dtype=np.int32) if x_shape else None
    wshape = np.array(w_shape, dtype=np.int32) if x_shape else None

    return pad, stride, dilation, xshape, wshape

def get_pad_tuple(padding, kernel):
    """Common code to get the pad option

    Parameters
    ----------
    padding : int or str
        Padding size, or ['VALID', 'SAME']

    kernel : tuple of int
        Conv kernel size

    Returns
    -------
    pad_top : int
        Padding size on top

    pad_left : int
        Padding size on left

    pad_down : int
        Padding size on down.

    pad_right : int
        Padding size on right.
    """
    # compute the padding size
    if isinstance(padding, (tuple, list)):
        if len(padding) == 2:
            pad_h = padding[0] * 2
            pad_w = padding[1] * 2
        elif len(padding) == 4:
            return  padding[0], padding[1], padding[2], padding[3]
        else:
            raise ValueError("Size of padding can only be 2 or 4")
    elif isinstance(padding, int):
        pad_h = pad_w = padding * 2
    elif padding == "VALID":
        pad_h = 0
        pad_w = 0
    elif padding == "SAME":
        pad_h = kernel[0] - 1
        pad_w = kernel[1] - 1
    else:
        raise ValueError("Unknown padding option %s" % padding)
    pad_top = (pad_h + 1) // 2
    pad_left = (pad_w + 1) // 2
    return pad_top, pad_left, pad_h - pad_top, pad_w - pad_left


def conv_output_shape(pad,
                      stride,
                      dilation,
                      x_shape,
                      w_shape,
                      data_dtype,
                      conv_dtype):
    """Get output shape of 2D or 3D convolution

    Paramters
    ---------
    tensor_format: int
        0: CUDNN_TENSOR_NCHW
        1: CUDNN_TENSOR_NHWC
        2: CUDNN_TENSOR_NCHW_VECT_C
    pad: int or list
        padding
    stride: int or list
        stride
    dilation: int or list
        dilation
    x_shape: list
        input shape
    w_shape: list
        weight shape
    data_dtype: str
        data type
    conv_dtype: str
        convolution type

    Returns
    -------
    oshape: list
        output shape
    """
    dims = len(x_shape)
    assert dims in (4, 5)
    pad = list(pad)
    #pad, stride, dilation, xshape, wshape = \
    #    _prepare_global_func_params(dims - 2, pad, stride, dilation, x_shape, w_shape)
    
    assert isinstance(stride, int) or len(stride) == 2
    assert isinstance(dilation, int) or len(dilation) == 2
    if isinstance(stride, int):
        stride_h = stride_w = stride
    else:
        stride_h, stride_w = stride

    if isinstance(dilation, int):
        dilation_h = dilation_w = dilation
    else:
        dilation_h, dilation_w = dilation

    if isinstance(pad, int):
        pad_h, pad_w = pad
    else:
        pad_h, pad_w = pad

    
    batch, in_channel, in_height, in_width = x_shape
    num_filter, channel, kernel_h, kernel_w = w_shape

    # compute the output shape
    dilated_kernel_h = (kernel_h - 1) * dilation_h + 1
    dilated_kernel_w = (kernel_w - 1) * dilation_w + 1
    pad_top, pad_left, pad_down, pad_right = get_pad_tuple(
        pad, (dilated_kernel_h, dilated_kernel_w))
    out_channel = num_filter
    out_height = ((in_height - dilated_kernel_h + pad_top + pad_down) // stride_h + 1)
    out_width = ((in_width - dilated_kernel_w + pad_left + pad_right) // stride_w + 1)

    #oshape = np.zeros((batch, out_channel, out_height, out_width), dtype="float32")
    return list([batch, out_channel, out_height, out_width])    

def add(x, y, is_input = False, is_output = False):
    x_shape  = list(x.shape)
    return _api.extern(
           x_shape, [x, y],
            lambda ins, outs: _intrin.call_packed(
                "tvm.contrib.nvdla.add",
                ins[0],
                ins[1],
                outs[0], is_input, is_output))


def relu(x, y, is_input = False, is_output = False):
    x_shape  = list(x.shape)
    return _api.extern(
           x_shape, [x, y],
            lambda ins, outs: _intrin.call_packed(
                "tvm.contrib.nvdla.relu",
                ins[0],
                ins[1],
                outs[0], is_input, is_output))


def conv_forward(x,
                 w,
                 pad,
                 stride,
                 dilation,
                 conv_dtype, is_input = False, is_output = False):
    dims = len(x.shape)
    assert dims in (4, 5)

    conv_dtype = x.dtype if conv_dtype is None else conv_dtype
    pad, stride, dilation, _, _ = \
        _prepare_global_func_params(dims - 2, pad, stride, dilation)

    oshape = conv_output_shape(
                               pad,
                               stride,
                               dilation,
                               list(x.shape),
                               list(w.shape),
                               x.dtype,
                               conv_dtype)
    if dims == 4:
        return _api.extern(
            oshape, [x, w],
            lambda ins, outs: _intrin.call_packed(
                "tvm.contrib.nvdla.conv2d",
                pad[0],
                pad[1],
                stride[0],
                stride[1],
                dilation[0],
                dilation[1],
                ins[0],
                ins[1],
                outs[0],
                conv_dtype, is_input, is_output))
    else:
        raise ValueError("Unsupport Dims: {}".format(dims))