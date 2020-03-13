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

def test_gemm_onnx():
    import onnx
# now you have super_resolution.onnx on disk
    onnx_model = onnx.load("/home/dev/resnet18v2.onnx")
    #print(onnx_model)
    data_shape = (1, 3, 224, 224)
    x = tvm.nd.array((np.random.uniform(size=data_shape)).astype("float32"))

    input_name = 'data'
    shape_dict = {input_name: x.shape}
    mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)

    func = mod["main"]

    from tvm.autotvm.graph_tuner.utils import has_multiple_inputs, get_direct_ancestor, get_in_nodes, \
    get_out_nodes, expr2graph, bind_inputs
    node_list = []
    node_dict = {}
    target_ops = []
    expr2graph(func, target_ops, node_dict, node_list)
    global global_stores
    global_stores["input_op"] = [node_list[99], node_list[0], node_list[1], node_list[2], node_list[3], node_list[4]]
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

    # Used for Debugging 
    # target = tvm.target.nvdla(options=["-debug"])
    # with relay.build_config(opt_level=0, disabled_pass=["AlterOpLayout", "SimplifyInference"]):
    #      _, lib, _ = relay.build(mod, target, params=params)
    # print(lib.get_source())
    # exit(0)


    target = tvm.target.nvdla(options=[])
    with relay.build_config(opt_level=0, disabled_pass=["AlterOpLayout", "SimplifyInference"]):
        graph, lib, params = relay.build(mod, target, params=params)

    module = tvm.contrib.graph_runtime.create(graph, lib, tvm.cpu())
    # set input and parameters
    x_data = np.random.uniform(size=data_shape).astype("float32")
    module.set_input(input_name, x_data)
    module.set_input(**params)
    # run
    module.run()



if __name__ == "__main__":
    test_gemm_onnx()