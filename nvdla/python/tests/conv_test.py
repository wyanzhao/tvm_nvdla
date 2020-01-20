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
"""Example code to do convolution."""

import numpy as np
import tvm
from tvm import autotvm
import topi
import topi.testing
from tvm.contrib.pickle_memoize import memoize
from topi.util import get_const_tuple

import tvm
import numpy as np
from tvm.contrib import util
import nvdla

def conv2d_nchw(Input, Filter, stride, padding, dilation):
    stride_h = stride_w = stride
    dilation_h = dilation_w = dilation
    batch, in_channel, in_height, in_width = Input.shape
    num_filter, channel, kernel_h, kernel_w = Filter.shape
    dilated_kernel_h = (kernel_h - 1) * dilation_h + 1
    dilated_kernel_w = (kernel_w - 1) * dilation_w + 1

    pad_h = padding
    pad_w = padding
    
    fout_height = (in_height + 2 * pad_h - kernel_h) // stride_h + 1
    fout_width = (in_width + 2 * pad_w - kernel_w) // stride_w + 1

    out_channel = num_filter

    rc = tvm.reduce_axis((0, in_channel), name='rc')
    ry = tvm.reduce_axis((0, kernel_h), name='ry')
    rx = tvm.reduce_axis((0, kernel_w), name='rx')
    return tvm.compute(
        (batch, out_channel, fout_height, fout_width),
        lambda nn, ff, yy, xx: tvm.sum(
            Input[nn, rc, yy * stride_h + ry * dilation_h,
                 xx * stride_w + rx * dilation_w].astype("float32") *
            Filter[ff, rc, ry, rx].astype("float32"),
            axis=[rc, ry, rx]), tag="conv2d")



def intrin_conv(op, input_name, weight_name, output_name):
    def intrin_func(ins, outs):
        ib = tvm.ir_builder.create()
        
        input_shape = ins[0].shape
        input_shape_len = len(input_shape)
        input_tensor = tvm.call_extern("handle", "AddFloatTensor",
                                str(input_name), input_shape_len, *(input_shape)
                                )
        ib.emit(tvm.call_extern("handle", "AddInputOp",
                                input_tensor
                                ))

        weight_shape = ins[1].shape
        weight_shape_len = len(weight_shape)
        ib.emit(tvm.call_extern("handle", "AddFloatWeightTensor",
                                str(weight_name), weight_shape_len, ins[1].data, *(weight_shape)
                                ))

        out_shape = outs[0].shape
        out_shape_len = len(out_shape)

        op = tvm.call_extern("handle", "AddConvOp",
                                str(input_name), str(weight_name))
        
        output_tensor = tvm.call_extern("handle", "AddFloatTensor",
                                str(output_name), out_shape_len, *(out_shape)
                                )
        
        ib.emit(tvm.call_extern("float", "AddOutput",
                                op, output_tensor
                                ))
        ib.emit(tvm.call_extern("float", "AddOutputOp",
                                output_name))
        ib.emit(tvm.call_extern("float", "Compile"))
        return ib.get()

    with tvm.build_config(offset_factor=1):
        return tvm.decl_tensor_intrin(op.op, intrin_func, binds={})

   

def verify_conv2d_nchw():
    A = tvm.placeholder((1, 1, 12, 8), name='A', dtype="float32")
    W = tvm.placeholder((13, 1, 1, 1), name='W', dtype="float32")

    a_shape = get_const_tuple(A.shape)
    w_shape = get_const_tuple(W.shape)
    dtype = A.dtype

    def get_ref_data():
        a_np = np.random.uniform(size=a_shape).astype(dtype)
        w_np = np.random.uniform(size=w_shape).astype(dtype)
        return a_np, w_np

    a_np, w_np = get_ref_data()
    w_np = np.load("/home/dev/W0.npy")

    C = conv2d_nchw(A, W, 1, 0, 1)
    
    s = tvm.create_schedule(C.op)
    print(tvm.lower(s, [A, W, C], simple_mode=True))
    intrin = intrin_conv(C, "data0", "weight0", "conv0")
    s[C].tensorize(C.op.axis[0], intrin)
    a = tvm.nd.array(a_np)
    w = tvm.nd.array(w_np)
    c = tvm.nd.array(np.zeros(get_const_tuple(C.shape), dtype=C.dtype))
    print(tvm.lower(s, [A, W, C], simple_mode=True))

    mhost = nvdla.build(s, [A, W, C], "c", name="CONV")
    
    temp = util.tempdir()
    path_dso = temp.relpath("temp.so")
    
    print(mhost.get_source())
    mhost.export_library(path_dso)
    m = tvm.module.load(path_dso)
    CONV = m['CONV']
    CONV(a, w, c)
    

def test_conv2d_nchw():
    verify_conv2d_nchw()


if __name__ == "__main__":
    test_conv2d_nchw()