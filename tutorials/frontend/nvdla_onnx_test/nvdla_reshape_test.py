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
"""Unit tests for graph partitioning."""
import os
import sys
import numpy as np

import mxnet as mx
import tvm
from tvm import relay
from tvm.contrib import util
import nvdla

from mxnet.gluon.model_zoo.vision import get_model
from tvm.contrib.download import download_testdata
from PIL import Image

from tvm.gloabal_value_store import global_stores  

def test_reshape_onnx():
    import onnx
# now you have super_resolution.onnx on disk
    onnx_model = onnx.load("/home/dev/Workspace/onnc-umbrella/src/single_layer_test/Reshape/5811_Reshape.onnx")

    x = tvm.nd.array((np.random.uniform(size=(1,1,1,2))).astype("float32"))
    y = tvm.nd.array((np.random.uniform(size=(1,1,1,1))).astype("float32"))
    z = tvm.nd.array((np.random.uniform(size=(1,1,1,2))).astype("float32"))

    input_name = 'IMAGE'
    weight_name = 'W0'
    init_name = "INIT1"
    shape_dict = {input_name: x.shape, weight_name: y.shape, init_name: z.shape}
    mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)
    print(mod)

    func = mod["main"]

    from tvm.autotvm.graph_tuner.utils import has_multiple_inputs, get_direct_ancestor, get_in_nodes, \
    get_out_nodes, expr2graph, bind_inputs
    node_list = []
    node_dict = {}
    target_ops = []
    expr2graph(func, target_ops, node_dict, node_list)
    global global_stores
    global_stores['op_infos'] = []
    global_stores["input_op"] = [node_list[3], node_list[0], node_list[1]]
    for x in node_list:
        node = {}
        if len(x['inputs']) != 0:
            node['name'] = x['op']
            node['node'] = x['node']
            node['input_shapes'] = []
            node['input_index'] = []
            node['op_shape'] = [int(x) for x in x['types'][0].shape]

            for y in x['inputs']:
                input_index = y[0]
                shape = [int(x) for x in node_list[input_index]['types'][0].shape]
                node['input_shapes'].append(shape)
                node['input_index'].append(input_index)
        else:
            node['name'] = x['op']
            node['node'] = x['node']
            node['op_shape'] = [int(x) for x in x['types'][0].shape]
            node['input_shapes'] = []
            node['input_index'] = []
        
        global_stores['op_infos'].append(node)
    global_stores["output_op"] = node_list[-1]

    #target = tvm.target.nvdla(options=["-debug"])
    target = tvm.target.nvdla(options=[])
    
    with relay.build_config(opt_level=0, disabled_pass=["AlterOpLayout", "SimplifyInference"]):
        graph, lib, params = relay.build(mod, target, params=params)
        
    #print(lib.get_source())

    module = tvm.contrib.graph_runtime.create(graph, lib, tvm.cpu())
    # set input and parameters
    x_data = np.random.uniform(size=(1,1,1,2)).astype("float32")
    a_data = np.array([[[[4,4]]]], dtype="float32")
    module.set_input("IMAGE", x_data)
    #module.set_input("INIT", a_data)
    module.set_input(**params)
    # run
    module.run()


def test_reshape_mxnet():
    data_shape = (1, 1, 1, 2)
    weight_shape=(1, 1, 1, 1)
    kernel_size = (1, 1)
    stride = (1, 1)
    num_filter = 1

    x = mx.sym.var("x", shape=data_shape, dtype="float32")
    a = mx.sym.var('a', shape=data_shape, dtype="float32")
    weight0 = mx.sym.var("weight0", shape=weight_shape, dtype="float32")
    conv = mx.sym.Convolution_v1(x, weight0, no_bias=True, kernel=kernel_size, num_filter=num_filter, stride=(1, 1))
    #reshape = mx.sym.reshape(conv, shape=(1, 2, 1, 1))
    reshape = mx.sym.reshape(conv, (1, 2, 1, 1))
    relu = mx.sym.relu(reshape)
    
    #reshape = mx.sym.BatchNorm(reshape)

    #reshape = mx.sym.reshape(reshape, shape=(1,1,1,2))
    # add = mx.sym.elemwise_add(conv, reshape)

    x_data = np.random.uniform(size=data_shape).astype("float32")
    a_data = np.array([[[[4,4]]]], dtype="float32")
    weight_data0 = np.array([[[[2]]]], dtype="float32")

    shape_dict = {"x": data_shape,"weight0": weight_shape}
    mod, params = relay.frontend.from_mxnet(relu, shape_dict)
    func = mod["main"]

    from tvm.autotvm.graph_tuner.utils import has_multiple_inputs, get_direct_ancestor, get_in_nodes, \
    get_out_nodes, expr2graph, bind_inputs
    node_list = []
    node_dict = {}
    target_ops = []
    expr2graph(func, target_ops, node_dict, node_list)
    global global_stores
    global_stores['op_infos'] = []
    global_stores["input_op"] = [node_list[2], node_list[0], node_list[1]]
    for x in node_list:
        node = {}
        if len(x['inputs']) != 0:
            node['name'] = x['op']
            node['node'] = x['node']
            node['input_shapes'] = []
            node['input_index'] = []
            node['op_shape'] = [int(x) for x in x['types'][0].shape]

            for y in x['inputs']:
                input_index = y[0]
                shape = [int(x) for x in node_list[input_index]['types'][0].shape]
                node['input_shapes'].append(shape)
                node['input_index'].append(input_index)
        else:
            node['name'] = x['op']
            node['node'] = x['node']
            node['op_shape'] = [int(x) for x in x['types'][0].shape]
            node['input_shapes'] = []
            node['input_index'] = []
        
        global_stores['op_infos'].append(node)
    global_stores["output_op"] = node_list[-1]

    target = tvm.target.nvdla(options=["-debug"])
    #target = tvm.target.nvdla()
    #target = "llvm"


    with relay.build_config(opt_level=0, disabled_pass=["AlterOpLayout", "SimplifyInference"]):
        graph, lib, params = relay.build(mod, target=target)
    print(lib.get_source())

    module = tvm.contrib.graph_runtime.create(graph, lib, tvm.cpu())

    # set input and parameters
    module.set_input("x", x_data)
    module.set_input("weight0", weight_data0)
    module.set_input(**params)
    # run
    module.run()


if __name__ == "__main__":
    test_reshape_onnx()
    #test_reshape_mxnet()