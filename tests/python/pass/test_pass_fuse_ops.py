# pylint: disable=attribute-defined-outside-init,invalid-name,protected-access,too-many-locals,too-many-statements
import numpy as np
import pytest
import mnm
from mnm.model import Conv2d
from mnm.model.trace import trace_mutate_attr
from mnm.testing import run_infer_type
import tvm
from tvm import relay


def t2m_param(param, ctx="cuda"):
    return mnm.ndarray(param, ctx=ctx)  # pylint: disable=unexpected-keyword-arg


def get_ctx_list():
    ret = ["cpu"]
    if mnm.build.with_cuda():
        ret.append("cuda")
    return ret


def randn(shape, *, ctx="cpu", dtype="float32"):
    x = np.random.randn(*shape)
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    assert list(x.shape) == list(shape)
    n_x = x.astype(dtype)
    m_x = mnm.array(n_x, ctx=ctx)
    m_x.requires_grad = True
    return m_x, n_x


def check(m_x, n_x, *, rtol=1e-5, atol=1e-5):
    m_x = m_x.asnumpy()
    np.testing.assert_allclose(m_x, n_x, rtol=rtol, atol=atol)


def test_fuse_simple():
    konst, _ = randn((1,), ctx="cpu")

    class Model(mnm.Model):
        def build(self):
            self.c = konst

        @mnm.model.trace
        def forward(self, x):
            y = mnm.add(x, self.c)
            y = mnm.relu(y)
            y = mnm.log(y)
            return y

    def expected(shape):
        add_op = mnm._ffi.op.GetOp("mnm.op.add")
        relu_op = mnm._ffi.op.GetOp("mnm.op.relu")
        log_op = mnm._ffi.op.GetOp("mnm.op.log")
        default = mnm._ffi.ir._make.Constant(mnm._core.value.IntValue(-114514))

        x = relay.var("p0", shape=shape)
        y = relay.var("p1", shape=(1,))
        z = relay.Call(add_op, [x, y, default, default])
        z = relay.Call(log_op, [relay.Call(relu_op, [z])])
        f1 = relay.Function([x, y], z)
        f1 = f1.with_attr("Primitive", tvm.tir.IntImm("int32", 1))
        x = relay.var("x", shape=shape)
        y = relay.var("c", shape=(1,))
        z = relay.var("a3")
        ret = relay.Let(z, relay.Call(f1, [x, y]), z)
        return relay.Function([x, y], ret)

    model = Model()
    m_x, _ = randn((10, 20), ctx="cpu")
    func_before = model._internal(m_x).func
    func_before = run_infer_type(func_before)
    func_after = mnm._ffi.pass_.FuseOps(func_before, 3)
    func_after = run_infer_type(func_after)
    func_expected = run_infer_type(expected((10, 20)))
    assert tvm.ir.structural_equal(func_after, func_expected)


def test_conv2d():
    rand, _ = randn((1,), ctx="cpu")

    class Model(mnm.Model):
        def build(self):
            self.c = rand
            self.conv1 = Conv2d(16, 16, kernel_size=(3, 3), padding=(1, 1), bias=False)
            self.conv2 = Conv2d(16, 16, kernel_size=(1, 1), padding=(0, 0), bias=False)
            self.conv3 = Conv2d(16, 16, kernel_size=(3, 3), padding=(1, 1), bias=False)

        @mnm.model.trace
        def forward(self, x):
            x = mnm.add(x, self.c)
            y = self.conv1(x)
            # this is the next dominator.
            y1 = mnm.add(y, self.c)
            y = mnm.add(y, y1)
            # second path
            z2 = self.conv2(y)
            z3 = self.conv3(y)
            # add can only be fused to z1
            z = mnm.add(z2, z3)
            return z

    def expected():
        konst0 = mnm._ffi.ir._make.Constant(mnm._core.value.IntValue(0))
        konst1 = mnm._ffi.ir._make.Constant(mnm._core.value.IntValue(1))
        konst_nchw = mnm._ffi.ir._make.Constant(mnm._core.value.StringValue("NCHW"))
        konst_oihw = mnm._ffi.ir._make.Constant(mnm._core.value.StringValue("OIHW"))
        default = mnm._ffi.ir._make.Constant(mnm._core.value.IntValue(-114514))
        add_op = mnm._ffi.op.GetOp("mnm.op.add")
        conv2d_op = mnm._ffi.op.GetOp("mnm.op.conv2d")

        # segment 0
        x = relay.var("p0", shape=(1, 16, 64, 64))
        y = relay.var("p1", shape=(1,))
        f0 = relay.Function([x, y], relay.Call(add_op, [x, y, default, default]))
        f0 = f0.with_attr("Primitive", tvm.tir.IntImm("int32", 1))

        # segment 1
        x = relay.var("p0", shape=(1, 16, 64, 64))
        w = relay.var("p1", shape=(16, 16, 3, 3))
        p2 = relay.var("p2", relay.TupleType((relay.TensorType((), "int64"),)))
        p3 = relay.var("p3", relay.TupleType((
            relay.TensorType((), "int64"), relay.TensorType((), "int64"),)))
        p4 = relay.var("p4", relay.TupleType((relay.TensorType((), "int64"),)))
        p5 = relay.var("p5", "int64")
        p6 = relay.var("p6", "int64")
        p7 = relay.var("p7", "int64")
        p8 = relay.var("p8", "int64")
        c = relay.var("c", shape=(1,))
        y = relay.Call(conv2d_op, [x, w, p2, p3, p4, p5, p6, p7, p8])
        y1 = relay.Call(add_op, [y, c, default, default])
        y = relay.Call(add_op, [y, y1, default, default])
        f1 = relay.Function([x, w, p2, p3, p4, p5, p6, p7, p8, c], y)
        f1 = f1.with_attr("Primitive", tvm.tir.IntImm("int32", 1))

        # segment 2
        x = relay.var("p0", shape=(1, 16, 64, 64))
        w = relay.var("p1", shape=(16, 16, 3, 3))
        p2 = relay.var("p2", relay.TupleType((relay.TensorType((), "int64"),)))
        p3 = relay.var("p3", relay.TupleType((
            relay.TensorType((), "int64"), relay.TensorType((), "int64"),)))
        p4 = relay.var("p4", relay.TupleType((relay.TensorType((), "int64"),)))
        p5 = relay.var("p5", "int64")
        p6 = relay.var("p6", "int64")
        p7 = relay.var("p7", "int64")
        p8 = relay.var("p8", "int64")
        y = relay.Call(conv2d_op, [x, w, p2, p3, p4, p5, p6, p7, p8])
        f2 = relay.Function([x, w, p2, p3, p4, p5, p6, p7, p8], y)
        f2 = f2.with_attr("Primitive", tvm.tir.IntImm("int32", 1))

        # segment 3
        x = relay.var("p0", shape=(1, 16, 64, 64))
        w = relay.var("p1", shape=(16, 16, 1, 1))
        p2 = relay.var("p2", relay.TupleType((relay.TensorType((), "int64"),)))
        p3 = relay.var("p3", relay.TupleType((
            relay.TensorType((), "int64"), relay.TensorType((), "int64"),)))
        p4 = relay.var("p4", relay.TupleType((relay.TensorType((), "int64"),)))
        p5 = relay.var("p5", "int64")
        p6 = relay.var("p6", "int64")
        p7 = relay.var("p7", "int64")
        p8 = relay.var("p8", "int64")
        offset = relay.var("offset", shape=(1, 16, 64, 64))
        y = relay.Call(conv2d_op, [x, w, p2, p3, p4, p5, p6, p7, p8])
        y = relay.Call(add_op, [y, offset, default, default])
        f3 = relay.Function([x, w, p2, p3, p4, p5, p6, p7, p8, offset], y)
        f3 = f3.with_attr("Primitive", tvm.tir.IntImm("int32", 1))

        # compose
        x = relay.var("x", shape=(1, 16, 64, 64))
        c = relay.var("c", shape=(1,))
        w1 = relay.var("conv1.w", shape=(16, 16, 3, 3))
        w2 = relay.var("conv2.w", shape=(16, 16, 1, 1))
        w3 = relay.var("conv3.w", shape=(16, 16, 3, 3))
        a1 = relay.var("a1")
        a4 = relay.var("a4")
        a6 = relay.var("a6")
        a7 = relay.var("a7")
        let3 = relay.Let(a7, relay.Call(f3, [a4, w2, konst1, konst0, konst1, konst1,
                                             konst_nchw, konst_oihw, konst_nchw, a6]), a7)
        let2 = relay.Let(a6, relay.Call(f2, [a4, w3, konst1, konst1, konst1, konst1,
                                             konst_nchw, konst_oihw, konst_nchw]), let3)
        let1 = relay.Let(a4, relay.Call(f1, [a1, w1, konst1, konst1, konst1, konst1,
                                             konst_nchw, konst_oihw, konst_nchw, c]), let2)
        let = relay.Let(a1, relay.Call(f0, [x, c]), let1)
        return relay.Function([x, c, w1, w2, w3], let)

    model = Model()
    m_x, _ = randn((1, 16, 64, 64), ctx="cpu")
    func_before = model._internal(m_x).func
    func_before = run_infer_type(func_before)
    func_after = mnm._ffi.pass_.FuseOps(func_before, 3)
    func_after = run_infer_type(func_after)
    func_expected = expected()
    func_expected = run_infer_type(func_expected)
    assert tvm.ir.structural_equal(func_after, func_expected)


def test_concatenate():
    """Test fusion case involving concat op and Tuple node"""
    rand, _ = randn((1,), ctx="cpu")

    class Model(mnm.Model):
        def build(self):
            self.c = rand

        @mnm.model.trace
        def forward(self, x):
            pooled = mnm.max_pool2d(x, kernel=(3, 3), stride=(1, 1), padding=1)
            concat = mnm.concatenate((pooled, x), axis=1)
            return mnm.add(concat, self.c)

    def expected(shape):
        max_pool2d_op = mnm._ffi.op.GetOp("mnm.op.max_pool2d")
        concat_op = mnm._ffi.op.GetOp("mnm.op.concatenate")
        add_op = mnm._ffi.op.GetOp("mnm.op.add")
        konst1 = mnm._ffi.ir._make.Constant(mnm._core.value.IntValue(1))
        konst3 = mnm._ffi.ir._make.Constant(mnm._core.value.IntValue(3))
        knchw = mnm._ffi.ir._make.Constant(mnm._core.value.StringValue("NCHW"))
        true = mnm._ffi.ir._make.Constant(mnm._core.value.BoolValue(True))
        false = mnm._ffi.ir._make.Constant(mnm._core.value.BoolValue(False))
        default = mnm._ffi.ir._make.Constant(mnm._core.value.IntValue(-114514))

        p0 = relay.var("p0", shape=shape)
        p1 = relay.var("p1", relay.TupleType((
            relay.TensorType((), "int64"), relay.TensorType((), "int64"),)))
        p2 = relay.var("p2", relay.TupleType((
            relay.TensorType((), "int64"), relay.TensorType((), "int64"),)))
        p3 = relay.var("p3", relay.TupleType((relay.TensorType((), "int64"),)))
        p4 = relay.var("p4", relay.TupleType((relay.TensorType((), "int64"),)))
        p5 = relay.var("p5", "bool")
        p6 = relay.var("p6", "bool")
        p7 = relay.var("p7", "int64")
        pooled = relay.Call(max_pool2d_op, [p0, p1, p2, p3, p4, p5, p6, p7])
        f1 = relay.Function([p0, p1, p2, p3, p4, p5, p6, p7], pooled)
        f1 = f1.with_attr("Primitive", tvm.tir.IntImm("int32", 1))

        p0 = relay.var("p0", shape=shape)
        p1 = relay.var("p0", shape=shape)
        p2 = relay.var("p2", shape=(1,))
        concat = relay.Call(concat_op, [relay.Tuple([p0, p1]), konst1])
        out = relay.Call(add_op, [concat, p2, default, default])
        f2 = relay.Function([p0, p1, p2], out)
        f2 = f2.with_attr("Primitive", tvm.tir.IntImm("int32", 1))

        x = relay.var("x", shape=shape)
        c = relay.var("c", shape=(1,))
        a1 = relay.var("a1")
        a4 = relay.var("a4")
        let2 = relay.Let(a4, relay.Call(f2, [a1, x, c]), a4)
        let1 = relay.Let(a1, relay.Call(f1, [x, konst3, konst1, konst1, konst1,
                                             false, true, knchw]), let2)
        return relay.Function([x, c], let1)

    model = Model()
    m_x, _ = randn((1, 16, 64, 64), ctx="cpu")
    before = model._internal(m_x).func
    before = run_infer_type(before)
    after = mnm._ffi.pass_.FuseOps(before, 3)
    after = run_infer_type(after)
    func_expected = run_infer_type(expected((1, 16, 64, 64)))
    assert tvm.ir.structural_equal(after, func_expected)


def test_tuple_root():
    """Test fusion case where Tuple node is the root in its group"""
    rand, _ = randn((1,), ctx="cpu")

    class Model(mnm.Model):
        def build(self):
            self.c = rand

        @mnm.model.trace
        def forward(self, x):
            pooled = mnm.max_pool2d(x, kernel=(3, 3), stride=(1, 1), padding=1)
            return (mnm.add(pooled, self.c), x)

    def expected(shape):
        max_pool2d_op = mnm._ffi.op.GetOp("mnm.op.max_pool2d")
        add_op = mnm._ffi.op.GetOp("mnm.op.add")
        konst1 = mnm._ffi.ir._make.Constant(mnm._core.value.IntValue(1))
        konst3 = mnm._ffi.ir._make.Constant(mnm._core.value.IntValue(3))
        knchw = mnm._ffi.ir._make.Constant(mnm._core.value.StringValue("NCHW"))
        true = mnm._ffi.ir._make.Constant(mnm._core.value.BoolValue(True))
        false = mnm._ffi.ir._make.Constant(mnm._core.value.BoolValue(False))
        default = mnm._ffi.ir._make.Constant(mnm._core.value.IntValue(-114514))

        p0 = relay.var("p0", shape=shape)
        p1 = relay.var("p1", relay.TupleType((
            relay.TensorType((), "int64"), relay.TensorType((), "int64"),)))
        p2 = relay.var("p2", relay.TupleType((
            relay.TensorType((), "int64"), relay.TensorType((), "int64"),)))
        p3 = relay.var("p3", relay.TupleType((relay.TensorType((), "int64"),)))
        p4 = relay.var("p4", relay.TupleType((relay.TensorType((), "int64"),)))
        p5 = relay.var("p5", "bool")
        p6 = relay.var("p6", "bool")
        p7 = relay.var("p7", "int64")
        c = relay.var("c", shape=(1,))

        pooled = relay.Call(max_pool2d_op, [p0, p1, p2, p3, p4, p5, p6, p7])
        out = relay.Call(add_op, [pooled, c, default, default])
        f = relay.Function([p0, p1, p2, p3, p4, p5, p6, p7, c], out)
        f = f.with_attr("Primitive", tvm.tir.IntImm("int32", 1))

        x = relay.var("x", shape=shape)
        c = relay.var("c", shape=(1,))
        a2 = relay.var("a2")
        a3 = relay.var("a3")
        let3 = relay.Let(a3, relay.Tuple([a2, x]), a3)
        let2 = relay.Let(
            a2, relay.Call(f, [x, konst3, konst1, konst1, konst1, false, true, knchw, c]), let3)
        return relay.Function([x, c], let2)

    model = Model()
    m_x, _ = randn((1, 16, 64, 64), ctx="cpu")
    before = model._internal(m_x).func
    before = run_infer_type(before)
    after = mnm._ffi.pass_.FuseOps(before, 3)
    after = run_infer_type(after)
    func_expected = expected((1, 16, 64, 64))
    func_expected = run_infer_type(func_expected)
    assert tvm.ir.structural_equal(after, func_expected)


def test_sgd():
    shape = [2, 3, 4]
    dtype = 'float32'
    ctx = "llvm"
    class Model(mnm.Model):
        def build(self):
            self.reset()

        def reset(self):
            self.x = mnm.array(np.random.randn(*shape).astype(dtype), ctx=ctx)

        @mnm.model.trace
        def forward(self, dy):  # pylint: disable=no-self-use
            y = mnm.relu(self.x)
            dx = mnm.relu_dx(self.x, y, dy)
            return y, dx

    class SGD(mnm.Model):
        def build(self, model, lr=0.1, mu=0.01):
            self.model = model
            self.lr = lr
            self.mu = mu
            self.reset()

        def reset(self):
            self.v = mnm.array(np.zeros(shape, dtype=dtype), ctx=ctx)
            self.model.reset()

        @mnm.model.trace
        def forward(self, dy):
            out = self.model(dy)
            y = out[0]
            dx = out[1]
            # update params
            sgd_out = mnm.sgd(self.model.x, dx, self.v, self.lr, self.mu)
            new_v = sgd_out[0]
            new_x = sgd_out[1]
            trace_mutate_attr(self.model, "x", new_x)
            trace_mutate_attr(self, "v", new_v)
            return y

    def expected():
        relu_op = mnm._ffi.op.GetOp("mnm.op.relu")
        relu_dx_op = mnm._ffi.op.GetOp("mnm.op.relu_dx")
        sgd_op = mnm._ffi.op.GetOp("mnm.op.sgd")
        default = mnm._ffi.ir._make.Constant(mnm._core.value.IntValue(-114514))

        x = relay.var("p0", shape=shape)
        dy = relay.var("p1", shape=shape)
        v = relay.var("p2", shape=shape)
        y = relay.Call(relu_op, [x])
        y1 = relay.Call(relu_dx_op, [x, y, dy])
        y2 = relay.Call(sgd_op, [x, y1, v, default, default])
        out = relay.Tuple([y, relay.TupleGetItem(y2, 1), relay.TupleGetItem(y2, 0)])
        f = relay.Function([x, dy, v], out)
        f = f.with_attr("Primitive", tvm.tir.IntImm("int32", 1))

        x = relay.var("model.x", shape=shape)
        dy = relay.var("dy", shape=shape)
        v = relay.var("v", shape=shape)
        a6 = relay.var("a6")
        let = relay.Let(a6, relay.Call(f, [x, dy, v]), a6)
        return relay.Function([dy, v, x], let)

    m_param, _ = randn(shape, ctx=ctx)
    n_v = np.zeros(shape, dtype=dtype)
    m_dy, _ = randn(shape, ctx=ctx)
    model = Model()
    sgd = SGD(model)
    model.x = m_param
    sgd.v = mnm.array(n_v, ctx=ctx)
    func = sgd._internal(m_dy).func
    func = run_infer_type(func)
    func = mnm._ffi.pass_.FuseOps(func, 3)
    func = run_infer_type(func)
    func_expected = expected()
    func_expected = run_infer_type(func_expected)
    assert tvm.ir.structural_equal(func, func_expected)


if __name__ == "__main__":
    pytest.main([__file__])
