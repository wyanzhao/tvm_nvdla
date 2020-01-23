#!/usr/bin/python3

import mxnet as mx
import numpy as np
from mxnet import ndarray as nd
from mxnet.contrib import onnx as onnx_mxnet
import logging
logging.basicConfig(level=logging.INFO)

input_shape = (1, 1, 2, 1)
weight_shape = (1, 1, 1, 1)

# Two placeholers are created with mx.sym.variable
A = mx.sym.Variable('A', shape=input_shape)
B = mx.sym.Variable("B", shape=weight_shape)

# The symbol is constructed using the '+' operator
CONV = mx.sym.Convolution(data=A, weight=B, kernel=(1, 1), num_filter=1)

RESHAPE = mx.sym.Reshape(data=CONV, shape=(1, 35))

print("Conv1 shape:{}".format(CONV.infer_shape()))
print("RESHAPE shape:{}".format(RESHAPE.infer_shape()))