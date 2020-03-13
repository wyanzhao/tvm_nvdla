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

def test_resnet18_mxnet():
    data_shape = (1, 64, 56, 56)
    weight_shape= (128, 64, 1, 1)
    kernel_size = (1, 1)
    num_filter =  128
    strides = (1, 1)

    x = mx.sym.var("x", shape=data_shape, dtype="float32")
    a = mx.sym.var('a', shape=data_shape, dtype="float32")
    weight0 = mx.sym.var("weight0", shape=weight_shape, dtype="float32")
    weight1 = mx.sym.var("weight1", shape=weight_shape, dtype="float32")
    conv1 = mx.sym.Convolution_v1(x, weight0, no_bias=True, kernel=kernel_size, num_filter=num_filter, stride=strides)
    conv2 = mx.sym.Convolution_v1(x, weight1, no_bias=True, kernel=kernel_size, num_filter=num_filter, stride=strides)
    add1 = mx.sym.elemwise_add(conv1, conv2)
    new_weight_shape = (128, 128, 1, 1)
    weight2 = mx.sym.var("weight2", shape=new_weight_shape, dtype="float32")
    weight3 =  mx.sym.var("weight3", shape=new_weight_shape, dtype="float32")
    conv3 = mx.sym.Convolution_v1(add1, weight2, no_bias=True, kernel=kernel_size, num_filter=128, stride=strides)
    conv4 = mx.sym.Convolution_v1(add1, weight3, no_bias=True, kernel=kernel_size, num_filter=128, stride=strides)
    add2 = mx.sym.elemwise_add(conv3, conv4)

    x_data = np.random.uniform(size=data_shape).astype("float32")
    weight_data = np.random.uniform(size=weight_shape).astype("float32")
    shape_dict = {"x": data_shape,"weight0": weight_shape, "weight1": weight_shape,
    "weight2": new_weight_shape, "weight3": new_weight_shape
    }
    mod, params = relay.frontend.from_mxnet(add2, shape_dict)
    print(mod)

    func = mod["main"]

    from tvm.autotvm.graph_tuner.utils import has_multiple_inputs, get_direct_ancestor, get_in_nodes, \
    get_out_nodes, expr2graph, bind_inputs
    node_list = []
    node_dict = {}
    target_ops = []
    expr2graph(func, target_ops, node_dict, node_list)
    global global_stores
    global_stores["input_op"] = [node_list[5], node_list[0], node_list[1]]
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
                if node_list[input_index]['op'] == 'null':
                    node['is_const'] = True
                else:
                    node['is_const'] = False
                node['input_shapes'].append(shape)
                node['input_index'].append(input_index)
            
            if global_stores['op_maps'].get(x['op']) != None:
                global_stores['op_maps'][x['op']].append(node)
            else:
                raise ValueError('Unsupport Op:{}'.format(x['op']))
        else:
            node['name'] = x['op']
            node['node'] = x['node']
            node['op_shape'] = [int(x) for x in x['types'][0].shape]
            node['input_shapes'] = []
            node['input_index'] = []
        
        global_stores['op_infos'] .append(node)
    global_stores["output_op"] = node_list[-1]

    target = tvm.target.nvdla(options=["-debug"])
    #target = "llvm"
    
    with relay.build_config(opt_level=0, disabled_pass=["AlterOpLayout", "SimplifyInference"]):
        graph, lib, params = relay.build(mod, target, params=params)
        
    print(lib.get_source())
    

    module = tvm.contrib.graph_runtime.create(graph, lib, tvm.cpu())
    # set input and parameters
    x_data = np.random.uniform(size=data_shape).astype("float32")
    module.set_input("x", x_data)
    module.set_input("weight0", weight_data)
    module.set_input(**params)
    # run
    module.run()



if __name__ == "__main__":
    test_resnet18_mxnet()
