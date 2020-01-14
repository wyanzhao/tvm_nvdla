import numpy
import onnx
from onnx import numpy_helper

tmp = numpy.load("/home/dev/Workspace/tvm/nvdla/python/W0-2.npy")
print("Array:{}".format(tmp))
print("Array shape:{}".format(tmp.shape))