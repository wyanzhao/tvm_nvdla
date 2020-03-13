
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
# pylint: disable=invalid-name,unused-variable,unused-argument,no-member
"""Conv2D schedule on x86"""

import logging
import re

import tvm
from tvm import autotvm
from tvm.autotvm.task.topi_integration import deserialize_args
from tvm.autotvm.task import get_config
from .. import generic, tag
from .. import nn
from ..util import simplify, get_const_tuple, get_const_int
from ..nn.conv2d import conv2d, conv2d_NCHWc, \
    conv2d_infer_layout, _get_workload as _get_conv2d_workload
from ..nn.depthwise_conv2d import _get_workload as _get_depthwise_conv2d_workload
from ..nn.pad import pad
from ..nn.util import get_pad_tuple

from .util import find_op_info

import operator

from tvm.contrib import nvdla

from tvm.gloabal_value_store import global_stores

@autotvm.register_topi_compute(conv2d, 'nvdla', ['direct'])
def _declaration_conv(cfg, data, kernel, strides, padding, dilation, layout, out_dtype):
    if layout == 'NCHW':
        if out_dtype is None:
            out_dtype = data.dtype
        assert isinstance(strides, int) or len(strides) == 2
        assert isinstance(dilation, int) or len(dilation) == 2
        assert isinstance(padding, int) or len(padding) == 4 or len(padding) == 2
        if isinstance(strides, int):
            stride_h = stride_w = strides
        else:
            stride_h, stride_w = strides

        if isinstance(dilation, int):
            dilation_h = dilation_w = dilation
        else:
            dilation_h, dilation_w = dilation

        if isinstance(padding, int):
            pad_w = pad_h = padding
        else:
            assert len(padding) == 2 or len(padding) == 4
            if len(padding) == 2:
                pad_h, pad_w = padding
            else:
                pad_h, pad_w, _, _ = padding

        batch, in_channel, in_height, in_width = data.shape
        num_filter, channel, kernel_h, kernel_w = kernel.shape
        dilated_kernel_h = (kernel_h - 1) * dilation_h + 1
        dilated_kernel_w = (kernel_w - 1) * dilation_w + 1


        fout_height = (in_height + 2 * pad_h - kernel_h) // stride_h + 1
        fout_width = (in_width + 2 * pad_w - kernel_w) // stride_w + 1

        out_channel = num_filter

        rc = tvm.reduce_axis((0, in_channel), name='rc')
        ry = tvm.reduce_axis((0, kernel_h), name='ry')
        rx = tvm.reduce_axis((0, kernel_w), name='rx')
        return tvm.compute(
            (batch, out_channel, fout_height, fout_width),
        lambda nn, ff, yy, xx: tvm.sum(
            data[nn, rc, yy * stride_h + ry * dilation_h,
                 xx * stride_w + rx * dilation_w].astype(out_dtype) *
            kernel[ff, rc, ry, rx].astype(out_dtype),
            axis=[rc, ry, rx]), tag="conv2d_nchw", attrs={"strides":strides, "padding": padding, "dilation": dilation})
    else:
        raise ValueError("not support this layout {} yet".format(layout))


def _intrin_conv(op_tensor, data_shape, kernel_shape, strides, padding, dilation, dtype, op_info,is_input = False, is_output = False):
    if isinstance(strides, int):
            stride_h = stride_w = strides
    else:
            stride_h, stride_w = strides

    if isinstance(dilation, int):
            dilation_h = dilation_w = dilation
    else:
            dilation_h, dilation_w = dilation

    if isinstance(padding, int):
            pad_w = pad_h = padding
    else:
        assert len(padding) == 2 or len(padding) == 4
        if len(padding) == 2:
            pad_h, pad_w = padding
        else:
            pad_h, pad_w, _, _ = padding

    batch, in_channel, in_height, in_width = data_shape
    num_filter, channel, kernel_h, kernel_w = kernel_shape
    out_channel = num_filter
    fout_height = (in_height + 2 * pad_h - kernel_h) // stride_h + 1
    fout_width = (in_width + 2 * pad_w - kernel_w) // stride_w + 1

    data = tvm.placeholder(data_shape, name='data', dtype=dtype)
    weight = tvm.placeholder(kernel_shape, name='weight', dtype=dtype)
    
    if int(stride_h) > 1:
        new_in_height = in_height - 1
        new_in_width = in_width - 1
    else:
        new_in_height = in_height
        new_in_width = in_width
    
    data_buffer = tvm.decl_buffer((tvm.var("e"), tvm.var("f"), tvm.var("g"), tvm.var("h")), dtype=data.dtype, strides=[tvm.var("a"), tvm.var("b"), tvm.var("c"), tvm.var("d")], name="data_buffer")
    #weight_buffer = tvm.decl_buffer(kernel_shape, dtype=weight.dtype, strides=strides, name="weight_buffer")

    rc = tvm.reduce_axis((0, in_channel), name='rc')
    ry = tvm.reduce_axis((0, kernel_h), name='ry')
    rx = tvm.reduce_axis((0, kernel_w), name='rx')

    conv_op = nn.conv2d(data, weight, strides=strides, padding=padding, dilation=dilation)
    # conv_op = tvm.compute(
    #         (batch, out_channel, fout_height, fout_width),
    #     lambda nn, ff, yy, xx: tvm.sum(
    #         #data[nn, rc, yy * stride_h + ry * dilation_h,
    #         #     xx * stride_w + rx * dilation_w].astype(data.dtype) *
    #         data[nn, rc, yy * stride_h + ry * dilation_h,
    #              xx * stride_w + rx * dilation_w].astype(data.dtype) *
    #         weight[ff, rc, ry, rx].astype(data.dtype),
    #         axis=[rc, ry, rx]), tag="conv2d_nchw", attrs={"strides": strides, "padding": padding, "dilation": dilation})
    

    def intrin_func(ins, outs):
        ib = tvm.ir_builder.create()

        global global_stores
        data = global_stores['op_infos'][op_info['input_index'][0]].get('node')
        kernel = global_stores['op_infos'][op_info['input_index'][1]].get('node')

        if is_input == True:
             ib.emit(tvm.call_extern("handle", "NvDlaInit", "Init"))
             input_shape = (batch, in_channel, in_height, in_width)
             input_shape_len = len(input_shape)
             assert input_shape_len == 4
             input_tensor = tvm.call_extern("handle", "AddFloatTensor",
                                 hash(data), input_shape_len, *(input_shape)
                                 )
             ib.emit(tvm.call_extern("handle", "AddInputOp",
                                     input_tensor
                                     ))
        weight_shape = ins[1].shape
        weight_shape_len = len(weight_shape)
        assert weight_shape_len == 4

        ib.emit(tvm.call_extern("handle", "AddFloatWeightTensorFromNumpy",
                                 hash(kernel), "Conv Weight",weight_shape_len, ins[1].data, *(weight_shape)
                                 ))

        out_shape = outs[0].shape
        out_shape_len = len(out_shape)
        assert out_shape_len == 4

        op = tvm.call_extern("handle", "AddConvOp",
                                 hash(data), hash(kernel))
        
        
        output_tensor = tvm.call_extern("handle", "AddFloatTensor",
                                 hash(op_info['node']), out_shape_len, *(out_shape)
                                 )
        
        ib.emit(tvm.call_extern("float", "AddOutput",
                                 op, output_tensor
                                 ))

        op_pointer = tvm.call_extern("handle", "GetOpPointer", hash(kernel))
        
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
                                 op_pointer, 4, pad_h, pad_w, pad_h, pad_w
                                 ))

        ib.emit(tvm.call_extern("handle", "SetConvStrides",
                                 op_pointer, 2, stride_h, stride_w
                                 ))
        
        if is_output == True:
             ib.emit(tvm.call_extern("float", "AddOutputOp",
                                 hash(op_info['node'])))
             ib.emit(tvm.call_extern("float", "NvDlaCompile"))

        return ib.get()

    with tvm.build_config(offset_factor=1):
        return tvm.decl_tensor_intrin(conv_op.op, intrin_func, binds={data: data_buffer})

"""
def _intrin_mul(input_tensors, op_tensor, dtype, is_output):
    assert len(input_tensors) == 2
    input_tensor1 = input_tensors[0]
    input_tensor2 = input_tensors[1]

    is_const = isinstance(input_tensor2.op, tvm.tensor.PlaceholderOp)

    input_tensor_id1 = hash(input_tensor1)
    input_tensor_id2 = hash(input_tensor2)
    op_tensor_id = hash(op_tensor)

    x = tvm.placeholder(input_tensor1.shape, dtype=dtype)
    y = tvm.placeholder(input_tensor2.shape, dtype=dtype)
    op = nn.topi.multiply(x, y)

    def intrin_func(ins, outs):
        ib = tvm.ir_builder.create()

        if is_const:
            weight_shape = ins[1].shape
            weight_shape_len = len(weight_shape)
            ib.emit(tvm.call_extern("handle", "AddFloatWeightTensorFromNumpy",
                                str(input_tensor_id2), weight_shape_len, ins[1].data, *(weight_shape)
                                ))

        out_shape = outs[0].shape
        out_shape_len = len(out_shape)
        #assert out_shape_len == 4

        op = tvm.call_extern("handle", "AddMulOp",
                                str(input_tensor_id1), str(input_tensor_id2)
                                )
        output_tensor = tvm.call_extern("handle", "AddFloatTensor",
                                str(op_tensor_id), out_shape_len, *(out_shape)
                                )        
        ib.emit(tvm.call_extern("float", "AddOutput",
                                op, output_tensor
                                ))

        if is_output:
            ib.emit(tvm.call_extern("float", "AddOutputOp", str(op_tensor_id)))
            ib.emit(tvm.call_extern("float", "NvDlaCompile"))
        return ib.get()

    with tvm.build_config(offset_factor=1):
        return tvm.decl_tensor_intrin(op.op, intrin_func)
"""


@autotvm.register_topi_schedule(generic.schedule_conv2d_nchw, 'nvdla', ['direct'])
def schedule_conv2d(cfg, outs):
    """Create schedule for tensors"""
    target = tvm.target.current_target(allow_none=False)
    outs = [outs] if isinstance(outs, tvm.tensor.Tensor) else outs
    s = tvm.create_schedule([x.op for x in outs])
    #return s

    s = tvm.create_schedule([x.op for x in outs])
    scheduled_ops = []

    global global_stores
    input_op = global_stores["input_op"][0]['op']
    param1 = global_stores["input_op"][1]["types"][0]
    param2 = global_stores["input_op"][2]["types"][0]
    output_op = global_stores["output_op"]['op']
    x = outs[0]

    def traverse(op):
        """Traverse operators from computation graph"""
        # inline all one-to-one-mapping operators except the last stage (output)
        """
        if tag.is_broadcast(op.tag):
            op_tensor = op.output(0)
            dtype = op_tensor.dtype
            input_tensors = op.input_tensors

            if op.name == "T_multiply":
                if op not in s.outputs:
                    intric = _intrin_mul(input_tensors, op_tensor, dtype, False)
                else:
                    intric = _intrin_mul(input_tensors, op_tensor, dtype, True)
                s[op].tensorize(op.axis[0], intric)
            else:
                raise ValueError("not support this op {} yet".format(op.name))     

            for tensor in op.input_tensors:
                if isinstance(tensor.op, tvm.tensor.ComputeOp) and tensor.op not in scheduled_ops:
                    traverse(tensor.op)
        """

        if op.tag == 'conv2d_nchw':
            conv_op = op.output(0)
            data, kernel = s[conv_op].op.input_tensors
            data_shape = data.shape
            kernel_shape = kernel.shape
            dtype = conv_op.dtype
            
            padding = op.attrs["padding"]
            dilation = op.attrs["dilation"]
            strides = op.attrs["strides"]

            is_input = False
            is_output = False

            data_shape = [int(x) for x in data_shape]
            param1_shape = [int(x) for x in param1.shape]
            kernel_shape = [int(x) for x in kernel_shape]
            param2_shape = [int(x) for x in param2.shape]

            op_shape = [int(x) for x in conv_op.shape]
            output_shape = [int(x) for x in global_stores['output_op']['types'][0].shape]

            if input_op == "conv2d" and data_shape == param1_shape and kernel_shape == param2_shape:
                is_input = True
            
            if output_op == "conv2d" and op_shape == output_shape:
                is_output = True

            op_info = find_op_info("conv2d", op_shape, s[conv_op].op.input_tensors)

            intric = _intrin_conv(conv_op, data_shape, kernel_shape, strides=strides, padding=padding,
                dilation=dilation, dtype=dtype, op_info=op_info, is_input=is_input, is_output=is_output)
            s[op].tensorize(op.axis[0], intric)
        else:
            raise ValueError("Unsupport Op:{}".format(op))

        scheduled_ops.append(op)
    
    traverse(outs[0].op)
    return s