import numpy
import onnx
from onnx import numpy_helper

# Preprocessing: create a Numpy array
numpy_array = numpy.array([-1.3, -2, 3.0], dtype=numpy.float32)
print('Original Numpy array:\n{}\n'.format(numpy_array))

# Convert the Numpy array to a TensorProto
tensor = numpy_helper.from_array(numpy_array)
print('TensorProto:\n{}'.format(tensor))

# Convert the TensorProto to a Numpy array
new_array = numpy_helper.to_array(tensor)
print('After round trip, Numpy array:\n{}\n'.format(new_array))

# Save the TensorProto
with open('tensor.pb', 'wb') as f:
    f.write(tensor.SerializeToString())

# Load a TensorProto
new_tensor = onnx.TensorProto()
with open('tensor.pb', 'rb') as f:
    new_tensor.ParseFromString(f.read())
print('After saving and loading, new TensorProto:\n{}'.format(new_tensor))