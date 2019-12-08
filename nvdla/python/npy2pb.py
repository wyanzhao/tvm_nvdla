#!/usr/bin/python3

import numpy
import onnx
from onnx import numpy_helper
import sys

if __name__ == "__main__":
    # Preprocessing: create a Numpy array
    if len(sys.argv) != 3:
        raise Exception("Give a input name and a output name")

    input_name = str(sys.argv[1])
    output_name = str(sys.argv[2])

    numpy_array = numpy.load(input_name)
    #print('Original Numpy array:\n{}\n'.format(numpy_array))
    print("Numpy array shape:{}".format(numpy_array.shape))

    # Convert the Numpy array to a TensorProto
    tensor = numpy_helper.from_array(numpy_array)
    #print('TensorProto:\n{}'.format(tensor))

    # Save the TensorProto
    with open(output_name, 'wb') as f:
        f.write(tensor.SerializeToString())
