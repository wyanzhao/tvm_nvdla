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
"""TVM operator fully connected compute."""
from __future__ import absolute_import
import tvm
from .. import tag

def gemm_default(data, weight, bias=None, transB = 0,out_dtype=None):
    """The default implementation of dense in topi.

    Parameters
    ----------
    data : tvm.Tensor
        2-D with shape [batch, in_dim]

    weight : tvm.Tensor
        2-D with shape [out_dim, in_dim]

    bias : tvm.Tensor, optional
        1-D with shape [out_dim]

    out_dtype : str
        The output type. This is used for mixed precision.

    Returns
    -------
    output : tvm.Tensor
        2-D with shape [batch, out_dim]
    """
    assert len(data.shape) == 2 and len(weight.shape) == 2, \
        "only support 2-dim dense"
    
    assert len(bias.shape) == 1 or len(bias.shape) == 2
    if out_dtype is None:
        out_dtype = data.dtype
    
    assert data.shape[1].value == weight.shape[0].value or data.shape[1].value == weight.shape[1].value

    #if len(bias.shape) == 1:
    #    assert bias.shape[0].value == weight.shape[1].value
    #else:
    #    assert bias.shape[0].value == data.shape[0].value and bias.shape[1].value == weight.shape[1].value
    if transB == 0: 
        M, K = data.shape
        N = weight.shape[1]
        k = tvm.reduce_axis((0, K), name='k')

        if len(bias.shape) == 1:
            matmul = tvm.compute((M, N), \
                             lambda i, j: tvm.sum((data[i, k].astype(out_dtype) * \
                                                  weight[i, j].astype(out_dtype)) + bias[j].astype(out_dtype), axis=k), \
                             name='T_gemm', tag='gemm')
        elif len(bias.shape) == 2:
            matmul = tvm.compute((M, N), \
                             lambda i, j: tvm.sum((data[i, k].astype(out_dtype) * \
                                                  weight[k, j].astype(out_dtype)) + bias[i, j].astype(out_dtype),  axis=k), \
                             name='T_gemm', tag='gemm')
        else:
            raise ValueError("Support Bias dimensions")
    else:
        M, K = data.shape
        N = weight.shape[0]
        k = tvm.reduce_axis((0, K), name='k')

        if len(bias.shape) == 1:
            matmul = tvm.compute((M, N), \
                             lambda i, j: tvm.sum((data[i, k].astype(out_dtype) * \
                                                  weight[j, k].astype(out_dtype)) + bias[j].astype(out_dtype), axis=k), \
                             name='T_gemm', tag='gemm')
        elif len(bias.shape) == 2:
            matmul = tvm.compute((M, N), \
                             lambda i, j: tvm.sum((data[i, k].astype(out_dtype) * \
                                                  weight[j, k].astype(out_dtype)) + bias[i, j].astype(out_dtype),  axis=k), \
                             name='T_gemm', tag='gemm')
        else:
            raise ValueError("Support Bias dimensions")
    return matmul


@tvm.target.override_native_generic_func("gemm")
def gemm(data, weight, bias=None, transB = 0,out_dtype=None):
    """Applies a linear transformation: :math:`Y = XW^T + b`.

    Parameters
    ----------
    data : tvm.Tensor
        2-D with shape [batch, in_dim]

    weight : tvm.Tensor
        2-D with shape [out_dim, in_dim]

    bias : tvm.Tensor, optional
        1-D with shape [out_dim]

    out_dtype : str
        The output type. This is used for mixed precision.

    Returns
    -------
    output : tvm.Tensor
        2-D with shape [batch, out_dim]
    """
    return gemm_default(data, weight, bias, transB, out_dtype)