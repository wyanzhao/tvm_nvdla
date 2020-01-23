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


def intrin_conv(op, input_name, weight_name, output_name, stride = 1, padding = 0, dilation = 1, is_input = False, is_output = False):
    stride_h = stride_w = stride
    dilation_h = dilation_w = dilation
    pad_h = padding
    pad_w = padding

    def intrin_func(ins, outs):
        ib = tvm.ir_builder.create()

        if is_input == True:
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
        ib.emit(tvm.call_extern("handle", "AddFloatWeightTensorFromNumpy",
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

        op_pointer = tvm.call_extern("handle", "GetOpPointer", str(input_name))
        
        # Set Conv Attributes
        ib.emit(tvm.call_extern("handle", "SetConvDilations",
                                op_pointer, 2, dilation_h, dilation_w
                                ))

        ib.emit(tvm.call_extern("handle", "SetConvGroup",
                                op_pointer, 1
                                ))

        ib.emit(tvm.call_extern("handle", "SetConvKernelShape",
                                op_pointer, 2, weight_shape[2], weight_shape[3]
                                ))
        
        ib.emit(tvm.call_extern("handle", "SetConvPads",
                                op_pointer, 2, pad_h, pad_w
                                ))

        ib.emit(tvm.call_extern("handle", "SetConvStrides",
                                op_pointer, 2, stride_h, stride_w
                                ))
        
        if is_output:
            ib.emit(tvm.call_extern("float", "AddOutputOp",
                                output_name))
            ib.emit(tvm.call_extern("float", "Compile"))

        
        return ib.get()

    with tvm.build_config(offset_factor=1):
        return tvm.decl_tensor_intrin(op.op, intrin_func, binds={})


def intrin_reshape(op, input_name1, input_name2, output_name, is_input=False, is_output=False):
    def intrin_func(ins, outs):
        ib = tvm.ir_builder.create()
        if is_input:
            input_shape_len = len(ins[0].shape)
            input_tensor = tvm.call_extern("handle", "AddFloatTensor",
                                    str(input_name1), input_shape_len, *(ins[0].shape)
                                    )
            ib.emit(tvm.call_extern("handle", "AddInputOp",
                                input_tensor
                                ))
        weight_shape = ins[0].shape
        weight_shape_len = len(weight_shape)
        ib.emit(tvm.call_extern("handle", "AddFloatWeightTensorFromNumpy",
                                str(input_name1), weight_shape_len, ins[0].data, *(weight_shape)
                                ))   

        out_shape = outs[0].shape
        out_shape_len = len(out_shape)

        
        ib.emit(tvm.call_extern("handle", "AddFloatWeightTensor",
                                str(input_name2), out_shape_len, *(out_shape)
                                ))


        op = tvm.call_extern("handle", "AddReshapeOp",
                                str(input_name1), str(input_name2)
                                )
        output_tensor = tvm.call_extern("handle", "AddFloatTensor",
                                str(output_name), out_shape_len, *(out_shape)
                                )
        
        ib.emit(tvm.call_extern("float", "AddOutput",
                                op, output_tensor
                                ))
        if is_output:
            ib.emit(tvm.call_extern("float", "AddOutputOp",
                                    output_name))
            ib.emit(tvm.call_extern("float", "Compile"))

        return ib.get()


    with tvm.build_config(offset_factor=1):
        return tvm.decl_tensor_intrin(op.op, intrin_func)


def intrin_relu(op, input_name, output_name, is_input=False, is_output=False):
    def intrin_func(ins, outs):
        ib = tvm.ir_builder.create()
        if is_input:
            input_shape_len = len(ins[0].shape)
            input_tensor = tvm.call_extern("handle", "AddFloatTensor",
                                    str(input_name), input_shape_len, *(ins[0].shape)
                                )
            ib.emit(tvm.call_extern("handle", "AddInputOp",
                                    input_tensor
                                ))

        out_shape = outs[0].shape
        out_shape_len = len(out_shape)

        op = tvm.call_extern("handle", "AddReluOp",
                                str(input_name)
                                )
        output_tensor = tvm.call_extern("handle", "AddFloatTensor",
                                str(output_name), out_shape_len, *(out_shape)
                                )        
        ib.emit(tvm.call_extern("float", "AddOutput",
                                op, output_tensor
                                ))

        if is_output:
            ib.emit(tvm.call_extern("float", "AddOutputOp",
                                output_name))
            ib.emit(tvm.call_extern("float", "Compile"))
        return ib.get()

    with tvm.build_config(offset_factor=1):
        return tvm.decl_tensor_intrin(op.op, intrin_func)

def intrin_add(op, input_name1, input_name2,output_name, is_input=False, is_output=False):
    def intrin_func(ins, outs):
        ib = tvm.ir_builder.create()
        if is_input:
            input_shape_len = len(ins[0].shape)
            input_tensor = tvm.call_extern("handle", "AddFloatTensor",
                                    str(input_name1), input_shape_len, *(ins[0].shape)
                                )
            ib.emit(tvm.call_extern("handle", "AddInputOp",
                                    input_tensor
                                ))

        out_shape = outs[0].shape
        out_shape_len = len(out_shape)

        op = tvm.call_extern("handle", "AddAddOp",
                                str(input_name1), str(input_name2)
                                )
        output_tensor = tvm.call_extern("handle", "AddFloatTensor",
                                str(output_name), out_shape_len, *(out_shape)
                                )        
        ib.emit(tvm.call_extern("float", "AddOutput",
                                op, output_tensor
                                ))

        if is_output:
            ib.emit(tvm.call_extern("float", "AddOutputOp",
                                output_name))
            ib.emit(tvm.call_extern("float", "Compile"))
        return ib.get()

    with tvm.build_config(offset_factor=1):
        return tvm.decl_tensor_intrin(op.op, intrin_func)



def intrin_maxpool(op, input_name, output_name, kernel, stride = 1, padding = 0, is_input = False, is_output = False):
    stride_h = stride_w = stride
    pad_h = padding
    pad_w = padding
    kernel_h, kernel_w = kernel

    def intrin_func(ins, outs):
        ib = tvm.ir_builder.create()

        if is_input == True:
            input_shape = ins[0].shape
            input_shape_len = len(input_shape)
            input_tensor = tvm.call_extern("handle", "AddFloatTensor",
                                str(input_name), input_shape_len, *(input_shape)
                                )
            ib.emit(tvm.call_extern("handle", "AddInputOp",
                                    input_tensor
                                    ))

        out_shape = outs[0].shape
        out_shape_len = len(out_shape)

        op = tvm.call_extern("handle", "AddMaxPoolOp",
                                str(input_name), 2, kernel_h, kernel_w)
        
        
        output_tensor = tvm.call_extern("handle", "AddFloatTensor",
                                str(output_name), out_shape_len, *(out_shape)
                                )
        
        ib.emit(tvm.call_extern("float", "AddOutput",
                                op, output_tensor
                                ))

        op_pointer = tvm.call_extern("handle", "GetOpPointer", str(input_name))
        
        # Set Conv Attributes
        ib.emit(tvm.call_extern("handle", "SetMaxPoolKernelShape",
                                op_pointer, 2, kernel_h, kernel_w
                                ))

        ib.emit(tvm.call_extern("handle", "SetMaxPoolPads",
                                op_pointer, 2, pad_h, pads_w
                                ))

        ib.emit(tvm.call_extern("handle", "SetMaxPoolStrides",
                                op_pointer, 2, stride_h, stride_w
                                ))
        
        if is_output:
            ib.emit(tvm.call_extern("float", "AddOutputOp",
                                output_name))
            ib.emit(tvm.call_extern("float", "Compile"))
        
        return ib.get()

    with tvm.build_config(offset_factor=1):
        return tvm.decl_tensor_intrin(op.op, intrin_func, binds={})

def relu(x):
    """Take relu of input x.

    Parameters
    ----------
    x : tvm.Tensor
        Input argument.

    Returns
    -------
    y : tvm.Tensor
        The result.
    """
    return tvm.compute(x.shape, lambda *i: tvm.max(x(*i), tvm.const(0, x.dtype)))

def add(x, y):
    return topi.add(x, y)

def maxpool(data,
         kernel,
         stride,
         padding,
         ceil_mode=False,
         layout="NCHW",
         count_include_pad=True):
    return topi.nn.pool(data, kernel, stride, padding, "max")

def reshape(a, new_shape):
    return topi.reshape(a, new_shape)

def lenet():
    dtype = "float32"
    
    A = tvm.placeholder((1, 1, 28, 28), name='data_0', dtype="float32")
    W1 = tvm.placeholder((32, 1, 5, 5), dtype="float32", name = "conv1_w0")

    # Conv1
    CONV1 = conv2d_nchw(A, W1, stride=1, padding=2, dilation=1)
    CONV1_BUFFER = tvm.placeholder((1, 32, 28, 28), dtype="float32")

    # Reshape1
    RESHAPE_DATA1 = tvm.placeholder((32, ), dtype="float32", name="reshape_data1")
    RESHAPE1 = reshape(RESHAPE_DATA1, (32, 1 , 1))

    # Add1
    ADD1 = add(CONV1, RESHAPE1)

    # RELU1
    RELU1 = relu(ADD1)

    # MAXPOOL1
    MAXPOOL1 = maxpool(RELU1, (2, 2), (2, 2), (0, 0, 0, 0))

    # CONV2
    W2 = tvm.placeholder((64, 32, 5, 5), dtype="float32", name="conv2_w0")
    CONV2 = conv2d_nchw(MAXPOOL1, W2, stride=1, padding=2, dilation=1)

    # RESHAPE2
    RESHAPE_DATA2 = tvm.placeholder((64, ), dtype="float32", name="reshape_data2")
    RESHAPE2 = reshape(RESHAPE_DATA2, (64, 1, 1))

    # Add2
    ADD2 = add(CONV2, RESHAPE2)

    # RELU2
    RELU2 = relu(ADD2)

    # MAXPOOL2
    MAXPOOL2 = maxpool(RELU2, (2, 2), (2, 2), (0, 0, 0, 0))

    W3 = tvm.placeholder((1024, 64, 7, 7), dtype="float32", name="conv3_w0")
    CONV3 = conv2d_nchw(MAXPOOL2, W3, stride=1, padding=0, dilation=1)

    # RESHAPE3
    RESHAPE_DATA3 = tvm.placeholder((1024, ), dtype="float32", name="reshape_data3")
    RESHAPE3 = reshape(RESHAPE_DATA3, (1, 1024, 1, 1))

    # ADD3
    ADD3 = add(CONV3, RESHAPE3)

    # RELU3
    RELU3 = relu(ADD3)

    # CONV4
    W4 = tvm.placeholder((10, 1024, 1, 1), dtype="float32", name="conv4_w0")
    CONV4 = conv2d_nchw(RELU3, W4, stride=1, padding=0, dilation=1)

    # RESHAPE4
    RESHAPE_DATA4 = tvm.placeholder((10, ), dtype="float32", name="reshape_data4")
    RESHAPE4 = reshape(RESHAPE_DATA4, (10, 1, 1))

    # ADD4
    ADD4 = add(CONV4, RESHAPE4)

    def get_lenet_data():
        a_np = np.random.uniform(size=A.shape).astype(dtype)
        w1_np = np.random.uniform(size=W1.shape).astype(dtype)
        w2_np = np.random.uniform(size=W2.shape).astype(dtype)
        w3_np = np.random.uniform(size=W3.shape).astype(dtype)
        w4_np = np.random.uniform(size=W4.shape).astype(dtype)
        reshape1_data = np.random.uniform(size=RESHAPE_DATA1.shape).astype(dtype)
        reshape2_data = np.random.uniform(size=RESHAPE_DATA2.shape).astype(dtype)
        reshape3_data = np.random.uniform(size=RESHAPE_DATA3.shape).astype(dtype)
        reshape4_data = np.random.uniform(size=RESHAPE_DATA4.shape).astype(dtype)
        
        return a_np, w1_np, w2_np, w3_np, w4_np, reshape1_data, reshape2_data, reshape3_data, reshape4_data
    
    # Before Schedule
    s = tvm.create_schedule(ADD4.op)
    #print(tvm.lower(s, [A, W1, W2, W3, W4, RESHAPE_DATA1, RESHAPE_DATA2, RESHAPE_DATA3, RESHAPE_DATA4], simple_mode=True))

    # Inject schedule
    conv1_intrin = intrin_conv(CONV1, "data_0", "conv1_w0", "conv1", stride=1, padding=2, dilation=1, is_input=True)
    s[CONV1].tensorize(CONV1.op.axis[0], conv1_intrin)

    reshape1_intrin = intrin_reshape(RESHAPE1, "reshape_data1", "reshape_dummy_1", "reshaped1")
    s[RESHAPE1].tensorize(RESHAPE1.op.axis[0], reshape1_intrin)
    s[RESHAPE1].compute_at(s[CONV1], CONV1.op.axis[0])
    #print(tvm.lower(s, [A, W1, W2, W3, W4, RESHAPE_DATA1, RESHAPE_DATA2, RESHAPE_DATA3, RESHAPE_DATA4], simple_mode=True))

    add1_intrin = intrin_add(ADD1, "conv1", "reshaped1", "bias_add_1")
    s[ADD1].tensorize(ADD1.op.axis[0], add1_intrin)

    relu1_intrin = intrin_relu(RELU1, "bias_add_1", "relu_1")
    s[RELU1].tensorize(RELU1.op.axis[0], relu1_intrin)

    maxpool1_intrin = intrin_maxpool(MAXPOOL1, "relu_1", "maxpool_1", kernel=2, stride=2, padding=0)
    s[MAXPOOL1].tensorize(MAXPOOL1.op.axis[0], maxpool1_intrin)

    conv2_intrin = intrin_conv(CONV2, "maxpool_1","conv2_w0", "conv2", stride=1, padding=2, dilation=1)
    s[CONV2].tensorize(CONV2.op.axis[0], conv2_intrin)




    print(tvm.lower(s, [A, W1, W2, W3, W4, RESHAPE_DATA1, RESHAPE_DATA2, RESHAPE_DATA3, RESHAPE_DATA4], simple_mode=True))


if __name__ == "__main__":
    lenet()