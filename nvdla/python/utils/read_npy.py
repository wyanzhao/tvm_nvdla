import numpy
import onnx
from onnx import numpy_helper

tmp = numpy.load("/home/dev/W0.npy")
print("Array:{}".format(tmp))
print("Array shape:{}".format(tmp.shape))