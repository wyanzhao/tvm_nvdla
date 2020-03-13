"""Conv2D schedule on x86"""

import logging
import re

import tvm
from tvm import autotvm
from tvm.autotvm.task.topi_integration import deserialize_args
from tvm.autotvm.task import get_config
from .. import generic, tag
from .. import nn
from ..nn.util import get_pad_tuple
from ..util import get_const_tuple
from .util import find_op_info

import operator

from tvm.contrib import nvdla

from tvm.gloabal_value_store import global_stores


def _intrin_global_average_pool(op_tensor, dtype, op_info, is_output = False):
    def intrin_func(ins, outs):
        ib = tvm.ir_builder.create()

        out_shape = outs[0].shape
        out_shape_len = len(out_shape)
        assert out_shape_len == 4
        
        global global_stores
        data = global_stores['op_infos'][op_info['input_index'][0]]['node']

        op = tvm.call_extern("handle", "AddGlobalAveragePoolOp",
                                 hash(data)
                                 )
        output_tensor = tvm.call_extern("handle", "AddFloatTensor",
                                 hash(op_info['node']), out_shape_len, *(out_shape)
                                 )        
        ib.emit(tvm.call_extern("float", "AddOutput",
                                 op, output_tensor
                                 ))

        if is_output:
             ib.emit(tvm.call_extern("float", "AddOutputOp", hash(op_info['node'])))
             ib.emit(tvm.call_extern("float", "NvDlaCompile"))
        return ib.get()

    with tvm.build_config(offset_factor=1):
        return tvm.decl_tensor_intrin(op_tensor.op, intrin_func)


def _intrin_max_pool2d(op_tensor, dtype, op_info, kernel_shape, padding, strides, is_output = False):
    def intrin_func(ins, outs):
        ib = tvm.ir_builder.create()

        if isinstance(strides, int):
            stride_h = stride_w = strides
        else:
            stride_h, stride_w = strides

        if isinstance(kernel_shape, int):
            kernel_shape_h = kernel_shape_w = kernel_shape
        else:
            kernel_shape_h, kernel_shape_w = kernel_shape

        if isinstance(padding, int):
            pad_w = pad_h = padding
        else:
            assert len(padding) == 4
            pad_1, pad_2, pad_3, pad_4 = padding

        out_shape = outs[0].shape
        out_shape_len = len(out_shape)
        #assert out_shape_len == 4
        
        global global_stores
        data = global_stores['op_infos'][op_info['input_index'][0]]['node']

        op = tvm.call_extern("handle", "AddMaxPoolOp",
                                 hash(data), len(kernel_shape), kernel_shape_h, kernel_shape_w
                                 )
        output_tensor = tvm.call_extern("handle", "AddFloatTensor",
                                 hash(op_info['node']), out_shape_len, *(out_shape)
                                 )        
        ib.emit(tvm.call_extern("float", "AddOutput",
                                 op, output_tensor
                                 ))
        
        op_pointer = tvm.call_extern("handle", "GetOpPointer", hash(data))
    

                # Set Conv Attributes
        ib.emit(tvm.call_extern("handle", "SetMaxPoolPads",
                                 op_pointer, 4, pad_1, pad_2, pad_3, pad_4
                                 ))
        
                # Set Conv Attributes
        ib.emit(tvm.call_extern("handle", "SetMaxPoolStrides",
                                 op_pointer, 2, stride_h, stride_w
                                 ))
        

        if is_output:
             ib.emit(tvm.call_extern("float", "AddOutputOp", hash(op_info['node'])))
             ib.emit(tvm.call_extern("float", "NvDlaCompile"))
        return ib.get()

    with tvm.build_config(offset_factor=1):
        return tvm.decl_tensor_intrin(op_tensor.op, intrin_func)


def _intrin_gemm(op_tensor, dtype, op_info, alpha=1.0, beta=1.0, trans_a = 0, trans_b = 0, is_output = False):
    def intrin_func(ins, outs):
        ib = tvm.ir_builder.create()
        
        global global_stores
        data = global_stores['op_infos'][op_info['input_index'][0]]['node']
        weight = global_stores['op_infos'][op_info['input_index'][1]]['node']
        bias = global_stores['op_infos'][op_info['input_index'][2]]['node']

        input1_shape = ins[1].shape
        input1_shape_len = len(input1_shape)
        assert input1_shape_len == 2

        input2_shape = ins[2].shape
        input2_shape_len = len(input2_shape)
        assert input2_shape_len == 2 or input2_shape_len == 1

        ib.emit(tvm.call_extern("handle", "AddFloatWeightTensorFromNumpy",
                                 hash(weight), "Gemm Weight",input1_shape_len, ins[1].data, *(input1_shape)
                                 ))
        ib.emit(tvm.call_extern("handle", "AddFloatWeightTensorFromNumpy",
                                 hash(bias), "Gemm Bias",input2_shape_len, ins[2].data, *(input2_shape)
                                 ))

        op = tvm.call_extern("handle", "AddGemmOp",
                                 hash(data), hash(weight), hash(bias)
                                 )

        out_shape = outs[0].shape
        out_shape_len = len(out_shape)
        
        output_tensor = tvm.call_extern("handle", "AddFloatTensor",
                                 hash(op_info['node']), out_shape_len, *(out_shape)
                                 )        
        ib.emit(tvm.call_extern("float", "AddOutput",
                                 op, output_tensor
                                 ))

        op_pointer = tvm.call_extern("handle", "GetOpPointer", hash(data))
        
        # Set Conv Attributes
        ib.emit(tvm.call_extern("handle", "SetGemmAlpha",
                                 op_pointer, alpha
                                 ))

                # Set Conv Attributes
        ib.emit(tvm.call_extern("handle", "SetGemmBeta",
                                 op_pointer, beta
                                 ))
        
                # Set Conv Attributes
        ib.emit(tvm.call_extern("handle", "SetGemmTransA",
                                 op_pointer, trans_a
                                 ))
        
                # Set Conv Attributes
        ib.emit(tvm.call_extern("handle", "SetGemmTransB",
                                 op_pointer, trans_b
                                 ))

        if is_output:
             ib.emit(tvm.call_extern("float", "AddOutputOp", hash(op_info['node'])))
             ib.emit(tvm.call_extern("float", "NvDlaCompile"))
        return ib.get()

    with tvm.build_config(offset_factor=1):
        return tvm.decl_tensor_intrin(op_tensor.op, intrin_func)


@generic.schedule_gemm.register(['nvdla'])
def schedule_gemm(outs, attrs):
    outs = [outs] if isinstance(outs, tvm.tensor.Tensor) else outs
    
    s = tvm.create_schedule([x.op for x in outs])
    scheduled_ops = []
    x = outs[0]
    op = x.op
    op_tensor = op.output(0)
    dtype = op_tensor.dtype
    input_tensors = op.input_tensors

    global global_stores
    output_op = global_stores["output_op"]['op']
    def traverse(op):
        """Internal traverse function"""
        # inline all one-to-one-mapping operators except the last stage (output)
        if tag.is_broadcast(op.tag):
            for tensor in op.input_tensors:
                if isinstance(tensor.op, tvm.tensor.ComputeOp) and tensor.op not in scheduled_ops:
                    traverse(tensor.op)
        # schedule pool
        elif op.name == ('T_gemm'):
            x = outs[0]
            is_output = False
        
            op_shape = [int(x) for x in x.shape]
            output_shape = [int(x) for x in global_stores['output_op']['types'][0].shape] 

            if output_op == "gemm" and op_shape == output_shape:
                is_output = True
            gemm_op = op.output(0)
            data = s[gemm_op].op.input_tensors

            op_info = find_op_info("gemm", op_shape, input_tensors)
        
            alpha = attrs["alpha"]
            beta = attrs["beta"]
            transA = attrs["transA"]
            transB = attrs['transB']

            intric = _intrin_gemm(gemm_op, dtype, op_info, alpha, beta, transA, transB, is_output=is_output)
            s[op].tensorize(op.axis[0], intric)
        else:
            raise RuntimeError("Unsupported operator: %s" % op.tag)

        scheduled_ops.append(op)

    traverse(outs[0].op)
    return s


@generic.schedule_pool.register(['nvdla'])
def schedule_maxpool_2d(outs, attrs):
    if attrs['layout'] != "NCHW":
        raise ValueError("Unsupport MaxPool2D Layout:{}".format(attrs['layout']))
    outs = [outs] if isinstance(outs, tvm.tensor.Tensor) else outs
    s = tvm.create_schedule([x.op for x in outs])
    scheduled_ops = []

    global global_stores
    output_op = global_stores["output_op"]['op']
    def traverse(op):
        """Internal traverse function"""
        # inline all one-to-one-mapping operators except the last stage (output)
        if tag.is_broadcast(op.tag):
            for tensor in op.input_tensors:
                if isinstance(tensor.op, tvm.tensor.ComputeOp) and tensor.op not in scheduled_ops:
                    traverse(tensor.op)
        # schedule pool
        elif op.tag == ('pool_max'):
            maxpool_op = op.output(0)
            dtype = maxpool_op.dtype
            input_tensors = op.input_tensors
            input0 = input_tensors[0]

            is_output = False
        
            op_shape = [int(x) for x in maxpool_op.shape]
            output_shape = [int(x) for x in global_stores['output_op']['types'][0].shape] 

            if output_op == "max_pool2d" and op_shape == output_shape:
                is_output = True

            op_info = find_op_info("max_pool2d", op_shape, input_tensors)

            #auto_pad = op.attrs["auto_pad"]
            kernel_shape = attrs['pool_size']
            padding = attrs['padding']
            strides = attrs["strides"]

            intric = _intrin_max_pool2d(maxpool_op, dtype, op_info, kernel_shape, padding, strides, is_output=is_output)
            s[op].tensorize(op.axis[0], intric)

        else:
            raise RuntimeError("Unsupported operator: %s" % op.tag)

        scheduled_ops.append(op)

    traverse(outs[0].op)
    return s


@generic.schedule_adaptive_pool.register(["nvdla"])
def schedule_adaptive_pool(outs):
    """Schedule for adaptive pool

    Parameters
    ----------
    outs: Array of Tensor
          The computation graph description of adaptive pool
          in the format of an array of tensors.

    Returns
    -------
    sch: Schedule
        The computation schedule for the op.
    """
    outs = [outs] if isinstance(outs, tvm.tensor.Tensor) else outs
    s = tvm.create_schedule([x.op for x in outs])
    scheduled_ops = []

    global global_stores
    output_op = global_stores["output_op"]['op']

    def traverse(op):
        """Internal traverse function"""
        # inline all one-to-one-mapping operators except the last stage (output)
        if tag.is_broadcast(op.tag):
            for tensor in op.input_tensors:
                if isinstance(tensor.op, tvm.tensor.ComputeOp) and tensor.op not in scheduled_ops:
                    traverse(tensor.op)
        # schedule pool
        elif op.tag == ('adaptive_pool_sum'):
            global_average_pool_op = op.output(0)
            dtype = global_average_pool_op.dtype
            input_tensors = op.input_tensors

            is_output = False
        
            op_shape = [int(x) for x in global_average_pool_op.shape]
            output_shape = [int(x) for x in global_stores['output_op']['types'][0].shape] 

            if output_op == "global_avg_pool2d" and op_shape == output_shape:
                is_output = True

            op_info = find_op_info("global_avg_pool2d", op_shape, input_tensors)

            intric = _intrin_global_average_pool(global_average_pool_op, dtype, op_info, is_output=is_output)
            s[op].tensorize(op.axis[0], intric)

        else:
            raise RuntimeError("Unsupported operator: %s" % op.tag)

        scheduled_ops.append(op)

    traverse(outs[0].op)
    return s