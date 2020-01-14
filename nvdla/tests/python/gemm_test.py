from __future__ import absolute_import, print_function

import tvm
import numpy as np

N, M, L = 1024, 512, 64
A = tvm.placeholder((N, L), name='A')
B = tvm.placeholder((M, L), name='B')
k = tvm.reduce_axis((0, L), name='k')
C = tvm.compute((N, M), lambda i, j:
                tvm.sum(A[i, k] * B[j, k], axis=k), name='C')
s = tvm.create_schedule(C.op)
print(tvm.lower(s, [A, B, C], simple_mode=True))

factor = 16
x, y = C.op.axis
z, = C.op.reduce_axis
yo, yi = s[C].split(y, factor=factor)
s[C].reorder(x, yo, yi, z)
print(tvm.lower(s, [A, B, C], simple_mode=True))

m = 16
l = 64
def intrin_gemv(m, l):
    a = tvm.placeholder((l,), name='a')
    b = tvm.placeholder((m, l), name='b')
    k = tvm.reduce_axis((0, l), name='k')
    c = tvm.compute((m,), lambda i: tvm.sum(a[k] * b[i, k], axis=k), name='c')
    Ab = tvm.decl_buffer(a.shape, a.dtype,
                         name="A",
                         offset_factor=1,
                         strides=[1])
    Bb = tvm.decl_buffer(b.shape, b.dtype,
                         name="B",
                         offset_factor=1,
                         strides=[tvm.var("s1"), 1])
    Cb = tvm.decl_buffer(c.shape, c.dtype,
                         name="C",
                         offset_factor=1,
                         strides=[1])
    def intrin_func(ins, outs):
        ib = tvm.ir_builder.create()
        aa, bb = ins
        cc = outs[0]
        ib.emit(tvm.call_extern("int32", "gemv_update",
                                cc.access_ptr("w"),
                                aa.access_ptr("r"),
                                bb.access_ptr("r"),
                                m, l, bb.strides[0]))
        return ib.get()
    with tvm.build_config(offset_factor=1):
        return tvm.decl_tensor_intrin(c.op, intrin_func, binds={a: Ab, b: Bb, c: Cb})

gemv = intrin_gemv(16, 64)

s[C].tensorize(yi, gemv)
print(tvm.lower(s, [A, B, C], simple_mode=True))

def gemv_impl():
    cc_code = """
      extern "C" int gemv_update(float *cc, float *aa, float *bb, int m, int l, int stride) {
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < l; ++j) {
                cc[i] += aa[j] * bb[i * stride + j];
            }
        }
        return 0;
      }
    """
    from tvm.contrib import util, clang
    temp = util.tempdir()
    ll_path = temp.relpath("temp.ll")
    # Create LLVM ir from c source code
    ll_code = clang.create_llvm(cc_code, output=ll_path)
    return ll_code

s[C].pragma(x, "import_llvm", gemv_impl())
print(tvm.lower(s, [A, B, C], simple_mode=True))

func = tvm.build(s, [A, B, C], target="llvm", name="gemv")

from topi.util import get_const_tuple
dtype = A.dtype
ctx = tvm.context("cpu", 0)
a = np.random.uniform(size=get_const_tuple(A.shape)).astype(dtype)
b = np.random.uniform(size=get_const_tuple(B.shape)).astype(dtype)
c = tvm.nd.array(np.zeros(get_const_tuple(C.shape), dtype=dtype), ctx)
func(tvm.nd.array(a, ctx), tvm.nd.array(b, ctx), c)
tvm.testing.assert_allclose(c.asnumpy(), np.dot(a, b.T), rtol=1e-3)