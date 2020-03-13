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
    block = get_model('resnet18_v2', pretrained=True)
    img_url = 'https://github.com/dmlc/mxnet.js/blob/master/data/cat.png?raw=true'
    img_name = 'cat.png'
    synset_url = ''.join(['https://gist.githubusercontent.com/zhreshold/',
                      '4d0b62f3d01426887599d4f7ede23ee5/raw/',
                      '596b27d23537e5a1b5751d2b0481ef172f58b539/',
                      'imagenet1000_clsid_to_human.txt'])
    synset_name = 'imagenet1000_clsid_to_human.txt'
    img_path = download_testdata(img_url, 'cat.png', module='data')
    synset_path = download_testdata(synset_url, synset_name, module='data')
    with open(synset_path) as f:
        synset = eval(f.read())
    image = Image.open(img_path).resize((224, 224))
    
    def transform_image(image):
        image = np.array(image) - np.array([123., 117., 104.])
        image /= np.array([58.395, 57.12, 57.375])
        image = image.transpose((2, 0, 1))
        image = image[np.newaxis, :]
        return image

    x = transform_image(image)
    shape_dict = {'data': x.shape}
    #print(block)

    mod, params = relay.frontend.from_mxnet(block, shape_dict)
    print(mod)
    return


def test_resnet18_onnx():
    import onnx
    model_url = ''.join(["https://s3.amazonaws.com/onnx-model-zoo/resnet/resnet18v2/resnet18v2.onnx"])
    model_path = download_testdata(model_url, 'resnet18v2.onnx', module='onnx')
# now you have super_resolution.onnx on disk
    onnx_model = onnx.load(model_path)

    img_url = 'https://github.com/dmlc/mxnet.js/blob/master/data/cat.png?raw=true'
    img_name = 'cat.png'
    synset_url = ''.join(['https://gist.githubusercontent.com/zhreshold/',
                      '4d0b62f3d01426887599d4f7ede23ee5/raw/',
                      '596b27d23537e5a1b5751d2b0481ef172f58b539/',
                      'imagenet1000_clsid_to_human.txt'])
    synset_name = 'imagenet1000_clsid_to_human.txt'
    img_path = download_testdata(img_url, 'cat.png', module='data')
    synset_path = download_testdata(synset_url, synset_name, module='data')
    with open(synset_path) as f:
        synset = eval(f.read())
    image = Image.open(img_path).resize((224, 224))
    
    def transform_image(image):
        image = np.array(image) - np.array([123., 117., 104.])
        image /= np.array([58.395, 57.12, 57.375])
        image = image.transpose((2, 0, 1))
        image = image[np.newaxis, :]
        return image

    x = transform_image(image)

    input_name = 'data'
    shape_dict = {input_name: x.shape}
    mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)
    print(mod)


if __name__ == "__main__":
    #test_resnet18_mxnet()
    test_resnet18_onnx()
