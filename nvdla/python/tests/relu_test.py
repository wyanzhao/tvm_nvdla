import tvm
import numpy as np
from tvm.contrib import util
import nvdla


def intrin_relu(op, input_name, output_name):

    def intrin_func(ins, outs):
        ib = tvm.ir_builder.create()

        input_shape_len = len(ins[0].shape)
        input_tensor = tvm.call_extern("handle", "AddFloatTensor",
                                str(input_name), input_shape_len, *(ins[0].shape)
                                )
        ib.emit(tvm.call_extern("handle", "AddInputOp",
                                input_tensor
                                ))

        out_shape = outs[0].shape
        out_shape_len = len(out_shape)

        op = tvm.call_extern("handle", "AddReluOp",
                                str(input_name)
                                )
        #ib.emit(op)
        output_tensor = tvm.call_extern("handle", "AddFloatTensor",
                                str(output_name), out_shape_len, *(out_shape)
                                )
        #ib.emit(output_tensor)
        
        ib.emit(tvm.call_extern("float", "AddOutput",
                                op, output_tensor
                                ))
        ib.emit(tvm.call_extern("float", "AddOutputOp",
                                output_name))
        ib.emit(tvm.call_extern("float", "Compile"))
        return ib.get()

    with tvm.build_config(offset_factor=1):
        return tvm.decl_tensor_intrin(op.op, intrin_func)

def relu(x):
    """Take relu of input x.

    Parameters
    ----------
    x : tvm.Tensor
        Input argument.

    Returns
    -------
    y : tvm.Tensor
        The result.
    """
    return tvm.compute(x.shape, lambda *i: tvm.max(x(*i), tvm.const(0, x.dtype)))

A = tvm.placeholder((1, 1, 3, 3), name='data0', dtype="int")
# A copy buffer
B = relu(A)
print(B)
s = tvm.create_schedule(B.op)

print(tvm.lower(s, [A, B], simple_mode=True))

intrin = intrin_relu(B, "data0", "activation0")
s[B].tensorize(B.op.axis[0], intrin)
#s[B].pragma(s[B].op.axis[0], "relu")
print(tvm.lower(s, [A, B], simple_mode=True))

mhost = nvdla.build(s, [A, B], "c", name="relu")

temp = util.tempdir()
path_dso = temp.relpath("temp.so")
source = mhost.get_source()
print(source)
mhost.export_library(path_dso)
m = tvm.module.load(path_dso)
relu = m['relu']
ctx = tvm.cpu(0)
inp = tvm.nd.array(np.ones((1, 1, 3, 3), dtype='float32'), ctx)
out = tvm.nd.array(np.empty((1, 1,3 ,3), dtype='float32'), ctx)
relu(inp, out)