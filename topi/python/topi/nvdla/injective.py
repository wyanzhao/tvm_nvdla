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
# pylint: disable=invalid-name
"""x86 declaration and schedules."""
from __future__ import absolute_import as _abs
import tvm
from .. import generic
from ..util import is_empty_shape

from .. import nn
from .. import cpp
from .util import find_op_info
from tvm import autotvm

from tvm.gloabal_value_store import global_stores

counter = 1

def _intrin_reshape(op_tensor, dtype, op_info, is_output = False):
    def intrin_func(ins, outs):
        global counter
        ib = tvm.ir_builder.create()

        int_shape = ins[0].shape
        int_shape_len = len(int_shape)

        out_shape = outs[0].shape
        out_shape_len = len(out_shape)

        global global_stores
        data = global_stores['op_infos'][op_info['input_index'][0]]['node']

        if op_info['is_const']:
            ib.emit(tvm.call_extern("handle", "AddFloatWeightTensorFromNumpy",
                                hash(data), "Reshape Weight", int_shape_len, ins[0].data, *(int_shape)
                                ))

        op = tvm.call_extern("handle", "AddReshapeOp",
                                 hash(data), out_shape_len, *(out_shape)
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


def _intrin_batch_norm(op_tensor, dtype, op_info, is_input = False, is_output = False): 
    def intrin_func(ins, outs):
        ib = tvm.ir_builder.create()

        global global_stores
        data = global_stores['op_infos'][op_info['input_index'][0]]['node']
        gamma = global_stores['op_infos'][op_info['input_index'][1]]['node']
        beta = global_stores['op_infos'][op_info['input_index'][2]]['node']
        mean = global_stores['op_infos'][op_info['input_index'][3]]['node']
        var = global_stores['op_infos'][op_info['input_index'][4]]['node']

        if is_input == True:
             ib.emit(tvm.call_extern("handle", "NvDlaInit", "Init"))
             input_shape = ins[0].shape
             input_shape_len = len(input_shape)
             assert input_shape_len == 4
             input_tensor = tvm.call_extern("handle", "AddFloatTensor",
                                 hash(data), input_shape_len, *(input_shape)
                                 )
             ib.emit(tvm.call_extern("handle", "AddInputOp",
                                     input_tensor
                                     ))

        gamma_shape = ins[1].shape
        gamma_shape_len = len(gamma_shape)
        ib.emit(tvm.call_extern("handle", "AddFloatWeightTensorFromNumpy",
                                hash(gamma), "BatchNorm Gamma",gamma_shape_len, ins[3].data, *(gamma_shape)
                                ))

        beta_shape = ins[2].shape
        beta_shape_len = len(beta_shape)
        ib.emit(tvm.call_extern("handle", "AddFloatWeightTensorFromNumpy",
                                hash(beta), "BatchNorm Beta",beta_shape_len, ins[4].data, *(beta_shape)
                                ))

        mean_shape = ins[3].shape
        mean_shape_len = len(mean_shape)
        ib.emit(tvm.call_extern("handle", "AddFloatWeightTensorFromNumpy",
                                hash(mean), "BatchNorm Mean",mean_shape_len, ins[1].data, *(mean_shape)
                                ))

        var_shape = ins[4].shape
        var_shape_len = len(var_shape)
        ib.emit(tvm.call_extern("handle", "AddFloatWeightTensorFromNumpy",
                                hash(var), "BatchNorm Var",var_shape_len, ins[2].data, *(var_shape)
                                ))

        out_shape = outs[0].shape
        out_shape_len = len(out_shape)
        assert out_shape_len == 4

        op = tvm.call_extern("handle", "AddBatchNormOp",
                                hash(data), hash(gamma), hash(beta), hash(mean), hash(var)
                                )

        output_tensor = tvm.call_extern("handle", "AddFloatTensor",
                                 hash(op_info['node']), out_shape_len, *(out_shape)
                                )        
        ib.emit(tvm.call_extern("float", "AddOutput",
                                op, output_tensor
                                ))

        if is_output:
            ib.emit(tvm.call_extern("float", "AddOutputOp",  hash(op_info['node'])))
            ib.emit(tvm.call_extern("float", "NvDlaCompile"))
        return ib.get()

    with tvm.build_config(offset_factor=1):
        return tvm.decl_tensor_intrin(op_tensor.op, intrin_func)


def _intrin_add(op_tensor, dtype, op_info, is_output=False):
    def intrin_func(ins, outs):
        ib = tvm.ir_builder.create()

        out_shape = outs[0].shape
        out_shape_len = len(out_shape)

        global global_stores
        #assert len(global_stores['op_infos'][op_info['input_index']]) == 2
        lhs = global_stores['op_infos'][op_info['input_index'][0]]['node']
        rhs = global_stores['op_infos'][op_info['input_index'][1]]['node']

        op = tvm.call_extern("handle", "AddAddOp",
                                hash(lhs), hash(rhs)
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


def _intrin_relu(op_tensor, dtype, op_info, is_output = False):
    def intrin_func(ins, outs):
        ib = tvm.ir_builder.create()

        out_shape = outs[0].shape
        out_shape_len = len(out_shape)
        
        global global_stores
        data = global_stores['op_infos'][op_info['input_index'][0]]['node']

        op = tvm.call_extern("handle", "AddReluOp",
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


def _intrin_identity(op_tensor, dtype, op_info, is_output = False):
    def intrin_func(ins, outs):
        ib = tvm.ir_builder.create()

        out_shape = outs[0].shape
        out_shape_len = len(out_shape)
        
        global global_stores
        data = global_stores['op_infos'][op_info['input_index'][0]]['node']

        if is_output:
             ib.emit(tvm.call_extern("float", "AddOutputOp", hash(data)))
             ib.emit(tvm.call_extern("float", "NvDlaCompile"))
        return ib.get()

    with tvm.build_config(offset_factor=1):
        return tvm.decl_tensor_intrin(op_tensor.op, intrin_func)


@generic.schedule_injective.register(["nvdla"])
def schedule_injective(outs):
    outs = [outs] if isinstance(outs, tvm.tensor.Tensor) else outs
    x = outs[0]
    s = tvm.create_schedule([x.op for x in outs])
    tvm.schedule.AutoInlineInjective(s)

    op = x.op
    op_tensor = op.output(0)
    dtype = op_tensor.dtype
    input_tensors = op.input_tensors

    global global_stores
    output_op = global_stores["output_op"]['op']

    if op.name == "T_relu":
        is_output = False

        op_shape = [int(x) for x in x.shape]
        output_shape = [int(x) for x in global_stores['output_op']['types'][0].shape]

        if output_op == "relu" and op_shape == output_shape:
           is_output = True

        op_info = find_op_info("relu", op_shape, input_tensors)

        intric = _intrin_relu(op_tensor, dtype, op_info, is_output=is_output)
        s[op].tensorize(op.axis[0], intric)

    elif op.name == "T_batch_norm":
        is_output = False
        is_input = False

        input_op = global_stores["input_op"][0]['op']
        param1 = global_stores["input_op"][1]["types"][0]
        param2 = global_stores["input_op"][2]["types"][0]
        param3 = global_stores["input_op"][3]["types"][0]
        param4 = global_stores["input_op"][4]["types"][0]
        param5 = global_stores["input_op"][5]["types"][0]

        param1_shape = [int(x) for x in param1.shape]
        param2_shape = [int(x) for x in param2.shape]
        param3_shape = [int(x) for x in param3.shape]
        param4_shape = [int(x) for x in param4.shape]
        param5_shape = [int(x) for x in param5.shape]


        input1, input2, input3, input4, input5 = s[op_tensor].op.input_tensors
        input1_shape = [int(x) for x in input1.shape]
        input2_shape = [int(x) for x in input2.shape]
        input3_shape = [int(x) for x in input3.shape]
        input4_shape = [int(x) for x in input4.shape]
        input5_shape = [int(x) for x in input5.shape]

        op_shape = [int(x) for x in x.shape]
        output_shape = [int(x) for x in global_stores['output_op']['types'][0].shape]

        if input_op == "batch_norm" and input1_shape == param1_shape and input2_shape == param2_shape and \
        input3_shape == param3_shape and input4_shape == param4_shape and input5_shape == param5_shape:
            is_input = True

        if output_op == "batch_norm" and op_shape == output_shape:
           is_output = True

        op_info = find_op_info("batch_norm", op_shape, input_tensors)
        
        intric = _intrin_batch_norm(op_tensor, dtype, op_info, is_input=is_input, is_output=is_output)
        s[op].tensorize(op.axis[0], intric)

    elif op.name == "T_add":
        is_output = False

        op_shape = [int(x) for x in x.shape]
        output_shape = [int(x) for x in global_stores['output_op']['types'][0].shape]

        if output_op == "add" and op_shape == output_shape:
            is_output = True
        
        add_op = op.output(0)

        op_info = find_op_info("add", op_shape, input_tensors)
        
        intric = _intrin_add(add_op, dtype, op_info, is_output=is_output)
        s[op].tensorize(op.axis[0], intric)

    elif op.name == "T_reshape":
        is_output = False
        
        op_shape = [int(x) for x in x.shape]
        output_shape = [int(x) for x in global_stores['output_op']['types'][0].shape] 

        if output_op == "reshape" and op_shape == output_shape:
            is_output = True
        reshape_op = op.output(0)
        data = s[reshape_op].op.input_tensors

        op_info = find_op_info("reshape", op_shape, input_tensors)

        intric = _intrin_reshape(reshape_op, dtype, op_info, is_output=is_output)
        s[op].tensorize(op.axis[0], intric)
    elif op.name == "T_identity":
        is_output = False
        
        identity_op_shape = [int(x) for x in x.shape]
        output_shape = [int(x) for x in global_stores['output_op']['types'][0].shape] 

        if output_op == "copy" and identity_op_shape == output_shape:
            is_output = True
        identity_op = op.output(0)
        data = s[identity_op].op.input_tensors

        op_info = None

        intric = _intrin_identity(identity_op, dtype, op_info, is_output=is_output)
        s[op].tensorize(op.axis[0], intric)
        
    else:
        if op.name != "Tensor":
            raise ValueError("Unsupport Operator:{}".format(op.name))

    return s


@generic.schedule_concatenate.register(["nvdla"])
def schedule_concatenate(outs):
    """X86 schedule for concatenate op.

    Parameters
    ----------
    outs: Array of Tensor
          The computation graph description of injective in the format
          of an array of tensors.

    Returns
    -------
    sch: Schedule
        The computation schedule for the op.
    """
    def vectorize(sch, tensor, vectorize_limit):
        """Internal vectorization function for concatenate."""
        inner_axis = s[tensor].op.axis[len(s[tensor].op.axis) - 1]
        inner_length = tensor.shape[len(tensor.shape) - 1].value
        if inner_length <= vectorize_limit:
            sch[tensor].vectorize(inner_axis)
        else:
            split_factor = 1
            for i in range(vectorize_limit, 1, -1):
                if inner_length % i == 0:
                    split_factor = i
                    break
            if split_factor > 1:
                _, inner_i = sch[tensor].split(inner_axis, split_factor)
                sch[tensor].vectorize(inner_i)

    outs = [outs] if isinstance(outs, tvm.tensor.Tensor) else outs
    x = outs[0]
    s = tvm.create_schedule([x.op for x in outs])
    tvm.schedule.AutoInlineInjective(s)
    if len(s[x].op.axis) >= 5:
        fused = s[x].fuse(s[x].op.axis[0], s[x].op.axis[1], s[x].op.axis[2])
        vectorize(s, x, 64)
        s[x].parallel(fused)
    elif len(s[x].op.axis) >= 3:
        fused = s[x].fuse(s[x].op.axis[0], s[x].op.axis[1])
        s[x].parallel(fused)
    else:
        s[x].parallel(s[x].op.axis[0])
    return s

schedule_elemwise = schedule_injective
schedule_broadcast = schedule_injective