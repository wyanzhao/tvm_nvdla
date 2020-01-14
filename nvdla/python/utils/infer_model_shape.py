#!/usr/bin/python3

import mxnet as mx
import numpy as np
from mxnet import ndarray as nd
from mxnet.contrib import onnx as onnx_mxnet
import logging
logging.basicConfig(level=logging.INFO)

input_shape = (1, 1, 28, 28)
weight_shape = (32, 1, 5, 5)

# Two placeholers are created with mx.sym.variable
A = mx.sym.Variable('A', shape=input_shape)
B = mx.sym.Variable("B", shape=weight_shape)

# The symbol is constructed using the '+' operator
CONV = mx.sym.Convolution(data=A, weight=B, kernel=(5, 5), num_filter=32, pad=(2,2))

print("Conv1 shape:{}".format(CONV.infer_shape()))

MAX_POOL = mx.sym.Pooling(data=CONV, kernel=(2, 2), pool_type='max', stride=(2, 2), pad=(0,0,0,0))
print("Maxpool1 shape:{}".format(MAX_POOL.infer_shape()))

input_shape = (1, 32, 14, 14)
weight_shape = (64, 32, 5, 5)

# Two placeholers are created with mx.sym.variable
A = mx.sym.Variable('A', shape=input_shape)
B = mx.sym.Variable("B", shape=weight_shape)

# The symbol is constructed using the '+' operator
CONV = mx.sym.Convolution(data=A, weight=B, kernel=(5, 5), num_filter=64, pad=(2,2))

print("Conv2 shape:{}".format(CONV.infer_shape()))
MAX_POOL = mx.sym.Pooling(data=CONV, kernel=(2, 2), pool_type='max', stride=(2, 2), pad=(0,0,0,0))
print("Maxpool2 shape:{}".format(MAX_POOL.infer_shape()))


input_shape = (1, 64, 7, 7)
weight_shape = (1024, 64, 7, 7)

# Two placeholers are created with mx.sym.variable
A = mx.sym.Variable('A', shape=input_shape)
B = mx.sym.Variable("B", shape=weight_shape)

# The symbol is constructed using the '+' operator
CONV = mx.sym.Convolution(data=A, weight=B, kernel=(7, 7), num_filter=1024)

print("Conv3 shape:{}".format(CONV.infer_shape()))

input_shape = (1, 1024, 1, 1)
weight_shape = (10, 1024, 1, 1)

# Two placeholers are created with mx.sym.variable
A = mx.sym.Variable('A', shape=input_shape)
B = mx.sym.Variable("B", shape=weight_shape)

# The symbol is constructed using the '+' operator
CONV = mx.sym.Convolution(data=A, weight=B, kernel=(1, 1), num_filter=10)


print("Conv4 shape:{}".format(CONV.infer_shape()))