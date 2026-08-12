# Owner(s): ["module: dsl-native-ops"]
#
# WIRING tests for the CuTeDSL reduction overrides. NUMERIC correctness (value /
# shape / dtype vs a reference) is NOT tested here -- the overrides are transparent
# replacements for aten reduction kernels, so their numerics are covered by the
# existing numpy-referenced OpInfo suites (test_reductions.py, test_ops.py) running
# with the override active. Duplicating that here would (a) be weaker than the
# numpy reference and (b) risk self-reference (computing a tolerance via an
# overridden op recurses into the kernel under test).
#
# This file covers only invariants of the OVERRIDE WIRING that OpInfo cannot
# express: that a supported call actually routes through our kernel (vs silent
# fallback), that the capability `cond` declines unsupported inputs and lets aten
# serve them, and that the kernels capture into CUDA graphs.

import functools
import math
import unittest

import torch
from torch.testing._internal.common_cuda import TEST_CUDA
from torch.testing._internal.common_utils import run_tests, skipIfNoCuteDSL, TestCase


def _disabled():
    return torch.backends.python_native.cutedsl.disabled()


@unittest.skipUnless(TEST_CUDA, "CUDA required")
@skipIfNoCuteDSL
class TestCuTeDSLReductionWiring(TestCase):
    def _fired_count(self, fn):
        # Count invocations of the dispatcher entry points our overrides funnel
        # through (reduce_dim / reduce_all single-output, reduce_dim2 two-output),
        # to prove a call routed to our kernel rather than aten.
        from torch._native.ops.reductions import kernel_general as kg

        names = ("reduce_dim", "reduce_dim2", "reduce_all")
        orig = {nm: getattr(kg, nm) for nm in names}
        n = [0]

        def wrap(f):
            def counting(*a, **k):
                n[0] += 1
                return f(*a, **k)

            return counting

        for nm in names:
            setattr(kg, nm, wrap(orig[nm]))
        try:
            fn()
        finally:
            for nm in names:
                setattr(kg, nm, orig[nm])
        return n[0]

    def test_supported_call_fires(self):
        # A supported call (CUDA, float, contiguous, valid dim) must route through
        # our kernel -- guards against a silently-all-fallback regression.
        x = torch.randn(128, 512, device="cuda")
        self.assertEqual(self._fired_count(lambda: torch.sum(x, dim=-1)), 1)
        self.assertEqual(self._fired_count(lambda: torch.mean(x, dim=-1)), 1)
        self.assertEqual(self._fired_count(lambda: torch.amax(x, dim=-1)), 1)
        # Group B: single-output index (argmax) and two-output (max.dim).
        self.assertEqual(self._fired_count(lambda: torch.argmax(x, dim=-1)), 1)
        self.assertEqual(self._fired_count(lambda: torch.max(x, dim=-1)), 1)
        # Group C: parameterized / non-float-output single reductions.
        self.assertEqual(self._fired_count(lambda: torch.var(x, dim=-1)), 1)
        self.assertEqual(
            self._fired_count(lambda: torch.linalg.vector_norm(x, dim=-1)), 1
        )
        self.assertEqual(self._fired_count(lambda: torch.count_nonzero(x, dim=-1)), 1)
        # Group D: two float-output reductions.
        self.assertEqual(self._fired_count(lambda: torch.var_mean(x, dim=-1)), 1)
        self.assertEqual(self._fired_count(lambda: torch.aminmax(x, dim=-1)), 1)

    def test_fp64_fires(self):
        # fp64 is a supported accumulator (not a fp32 cast), so a fp64 call must
        # route through our kernel rather than fall back to aten.
        x = torch.randn(128, 512, device="cuda", dtype=torch.float64)
        self.assertEqual(self._fired_count(lambda: torch.sum(x, dim=-1)), 1)
        self.assertEqual(self._fired_count(lambda: torch.amax(x, dim=-1)), 1)

    def test_unsupported_dtype_falls_back(self):
        # Complex is the only input dtype left outside the supported set -> must NOT hit
        # our kernel, and must still be correct (served by aten).
        for dtype in (torch.complex64, torch.complex128):
            xi = torch.ones(64, 64, device="cuda", dtype=dtype)
            with self.subTest(dtype=dtype):
                self.assertEqual(self._fired_count(lambda: torch.sum(xi, dim=-1)), 0)
                with _disabled():
                    ref = torch.sum(xi, dim=-1)
                self.assertEqual(torch.sum(xi, dim=-1), ref)

    def test_narrow_int_and_bool_fire(self):
        # int8/int16/uint8 and bool ARE served. aten PROMOTES the narrow integers for a
        # sum-like reduction (int8 in -> int64 out) and keeps the input width for a value
        # reduction; bool travels as its uint8 storage bytes. exact_dtype matters most
        # here: uint8 is the one dtype whose all/any result is NOT bool (aten's legacy
        # ByteTensor rule), and getting that wrong is invisible to a value-only check.
        for dtype in (torch.int8, torch.int16, torch.uint8, torch.bool):
            x = torch.randint(0, 3, (128, 512), device="cuda").to(dtype)
            with self.subTest(dtype=dtype):
                for fn in (
                    lambda t: torch.sum(t, dim=-1),
                    lambda t: torch.amax(t, dim=-1),
                    lambda t: torch.max(t, dim=-1),
                    lambda t: torch.aminmax(t, dim=-1),
                    lambda t: torch.all(t, dim=-1),
                    lambda t: torch.any(t, dim=-1),
                    lambda t: torch.count_nonzero(t, dim=-1),
                ):
                    self.assertEqual(self._fired_count(lambda: fn(x)), 1)
                    with _disabled():
                        ref = fn(x)
                    got = fn(x)
                    for g, r in zip(
                        got if isinstance(got, tuple) else (got,),
                        ref if isinstance(ref, tuple) else (ref,),
                    ):
                        self.assertEqual(g.dtype, r.dtype)
                        self.assertEqual(g, r, exact_dtype=True)

    def test_empty_reduction_is_served_or_declined_by_identity(self):
        # An empty reduction has nothing to launch a kernel over, so the result is either an
        # empty tensor (a KEPT extent is zero -- true for every op, even amax) or the op's
        # IDENTITY over the kept shape (only for an op that HAS one). The max/min family has
        # none and aten raises for it, so those must still decline -- serving them would
        # return a value where aten errors.
        empty_out = torch.randn(0, 5, device="cuda")
        empty_axis = torch.randn(5, 0, device="cuda")
        for fn in (
            lambda t: t.sum(dim=1),
            lambda t: t.mean(dim=1),
            lambda t: torch.prod(t, 1),
            lambda t: t.all(dim=1),
            lambda t: torch.count_nonzero(t, dim=1),
            lambda t: torch.linalg.vector_norm(t, dim=1),
        ):
            for x in (empty_out, empty_axis):
                with self.subTest(fn=fn, shape=tuple(x.shape)):
                    with _disabled():
                        ref = fn(x)
                    got = fn(x)
                    self.assertEqual(got.shape, ref.shape)
                    self.assertEqual(got.dtype, ref.dtype)
                    self.assertEqual(got, ref, exact_dtype=True)
        # No identity + a non-empty output -> aten must raise, so we must not answer. aten
        # reports this as an IndexError (TORCH_CHECK_INDEX on the reduced dim).
        for fn in (torch.amax, torch.amin, torch.argmax, torch.aminmax):
            with self.subTest(fn=fn):
                with self.assertRaises((RuntimeError, IndexError)):
                    fn(empty_axis, dim=1)
        # ... but the same ops over an EMPTY output are served, since no identity is needed.
        for fn in (torch.amax, torch.amin, torch.argmax):
            with self.subTest(fn=fn, empty_out=True):
                self.assertEqual(fn(empty_out, dim=1).shape, (0,))

    def test_argmax_declines_bool(self):
        # aten REFUSES a bool input to arg* ("argmax(): does not support bool input"),
        # unlike max.dim, so serving it would return an index where aten raises.
        x = torch.ones(64, 64, device="cuda", dtype=torch.bool)
        for fn in (torch.argmax, torch.argmin):
            with self.assertRaisesRegex(RuntimeError, "does not support bool"):
                fn(x, dim=-1)

    def test_full_reduce_overloads_fire(self):
        # aten::prod / max / min / all / any / count_nonzero each have their own CUDA
        # dispatch (only aten::sum decomposes onto its dim overload), so each needs a
        # registration of its own or the most idiomatic spelling of the reduction --
        # torch.prod(x), torch.max(x) -- silently falls back.
        x = torch.randint(0, 3, (64, 64), device="cuda")
        xf = torch.rand(64, 8, device="cuda") + 0.5
        for fn, t in (
            (torch.prod, xf),
            (torch.max, x),
            (torch.min, x),
            (torch.all, x),
            (torch.any, x),
            (torch.count_nonzero, x),
            (lambda a: torch.all(a, dim=(0, 1)), x),
            (lambda a: torch.any(a, dim=(1,)), x),
        ):
            self.assertGreaterEqual(self._fired_count(lambda: fn(t)), 1)
            with _disabled():
                ref = fn(t)
            self.assertEqual(fn(t), ref, exact_dtype=True)

    def test_int32_int64_fire(self):
        # int32/int64 ARE served: the accumulator is Int64 (so an int32 sum does not wrap
        # where aten's would not) and aten's integral promotion is honoured -- sum/prod
        # return int64 for an int32 input, while amax/max.dim keep int32.
        for dtype in (torch.int32, torch.int64):
            x = torch.randint(-8, 9, (128, 512), device="cuda", dtype=dtype)
            with self.subTest(dtype=dtype):
                for fn in (
                    lambda t: torch.sum(t, dim=-1),
                    lambda t: torch.amax(t, dim=-1),
                    lambda t: torch.argmax(t, dim=-1),
                    lambda t: torch.max(t, dim=-1),
                    lambda t: torch.aminmax(t, dim=-1),
                    lambda t: torch.count_nonzero(t, dim=-1),
                ):
                    self.assertEqual(self._fired_count(lambda: fn(x)), 1)
                    with _disabled():
                        ref = fn(x)
                    got = fn(x)
                    for g, r in zip(
                        got if isinstance(got, tuple) else (got,),
                        ref if isinstance(ref, tuple) else (ref,),
                    ):
                        self.assertEqual(g.dtype, r.dtype)
                        self.assertEqual(g, r, exact_dtype=True)

    def test_int32_sum_does_not_wrap_at_2_31(self):
        # The accumulator must be Int64: 512 * 2**24 overflows int32, and aten's integral
        # sum does not wrap there, so neither may ours. (An Int32 acc silently wrapped.)
        x = torch.full((4, 512), 1 << 24, device="cuda", dtype=torch.int32)
        self.assertEqual(self._fired_count(lambda: torch.sum(x, dim=-1)), 1)
        got = torch.sum(x, dim=-1)
        self.assertEqual(got.dtype, torch.int64)
        self.assertEqual(got, torch.full((4,), 512 * (1 << 24), device="cuda"))

    @staticmethod
    def _compiled_kernel_count():
        # Every compiled reduction kernel lands in exactly one of these caches, keyed on its
        # compile signature -- so len() IS the number of distinct kernels built so far.
        from torch._native.ops.reductions import (
            kernel_coltile,
            kernel_general,
            kernel_rowtile,
            kernel_xcta,
        )

        caches = (
            kernel_general._COMPILE_CACHE,
            kernel_rowtile._CACHE,
            kernel_coltile._CACHE,
            kernel_xcta._PLAN,
        )
        return sum(len(c) for c in caches), caches

    def test_kernel_count_does_not_scale_with_shape(self):
        # GUARD. The whole design rests on compiling O(op x dtype x structure) kernels, not
        # O(distinct shapes): the geometry is passed as runtime args and the fold loops are
        # ROLLED. A size-derived const_expr (a range_constexpr over a count, say) silently
        # breaks that and shows up first as compile time, not as a wrong answer -- so assert
        # it directly. The bound is RELATIVE (does the count saturate?) rather than a magic
        # number, so it survives adding kernels or options.
        #
        # These N all share a vec class (all multiples of 8) and sit in one bucket rung, so a
        # correctly-parameterized stack compiles the SAME kernels for two of them as for ten.
        #
        # This covers the DEFAULT (rolled) fold orders, which is every order the stack picks on its
        # own. An order that fixes its add DAG at compile time cannot satisfy this and is not meant
        # to -- it keys on N by construction, in exchange for a reproducible bit pattern -- so it is
        # opt-in and off here (see kernel_rowtile's inner-tree order).
        few = [4096, 4104]
        many = [4096, 4104, 4112, 4120, 4128, 4136, 4144, 4152, 4160, 4168]

        def run(sizes):
            _, caches = self._compiled_kernel_count()
            for c in caches:
                c.clear()
            for n in sizes:
                x = torch.randn(512, n, device="cuda")
                torch.sum(x, dim=-1)
                torch.amax(x, dim=-1)
                # column path too: its split factor and stage-2 mapping are shape-derived
                torch.sum(x, dim=0)
                del x
            return self._compiled_kernel_count()[0]

        n_few = run(few)
        n_many = run(many)
        self.assertEqual(
            n_many,
            n_few,
            f"compiled kernels grew from {n_few} (2 shapes) to {n_many} (10 shapes): a "
            f"size-derived const_expr has crept back in",
        )

    def test_int_extremes_and_overflow_match_aten(self):
        # The accumulator identities are INT64_MIN/MAX (traits._neg_id/_pos_id), so an
        # input AT those extremes must still win/lose correctly, and an overflowing int64
        # sum/prod must WRAP exactly as aten's does rather than saturate.
        i32min, i32max = -(2**31), 2**31 - 1
        z = torch.full((4, 64), 7, device="cuda", dtype=torch.int32)
        z[:, 3], z[:, 9] = i32min, i32max
        w = torch.full((4, 32), -5, device="cuda", dtype=torch.int64)
        w[:, 1], w[:, 2] = 2**63 - 1, -(2**63)
        cases = [
            (z, lambda t: t.amax(dim=1)),
            (z, lambda t: t.amin(dim=1)),
            (z, lambda t: t.argmax(dim=1)),
            (z, lambda t: t.argmin(dim=1)),
            (w, lambda t: t.amax(dim=1)),
            (w, lambda t: t.amin(dim=1)),
        ]
        # overflow: 8 * 2**62 and (2**20)**8 both wrap in int64
        big = functools.partial(torch.full, (4, 8), device="cuda", dtype=torch.int64)
        cases += [
            (big(1 << 62), lambda t: t.sum(dim=1)),
            (big(1 << 20), lambda t: t.prod(dim=1)),
        ]
        for i, (x, fn) in enumerate(cases):
            with self.subTest(case=i):
                self.assertEqual(self._fired_count(lambda: fn(x)), 1)
                with _disabled():
                    ref = fn(x)
                self.assertEqual(fn(x), ref, exact_dtype=True)

    def test_float_only_reductions_decline_ints(self):
        # mean/var/std/vector_norm have no integer semantics in aten (it raises), so their
        # conds must stay float-only rather than serve a call aten would reject.
        xi = torch.randint(1, 9, (64, 64), device="cuda", dtype=torch.int32)
        for fn in (
            lambda t: torch.mean(t, dim=-1),
            lambda t: torch.var(t, dim=-1),
            lambda t: torch.std(t, dim=-1),
            lambda t: torch.linalg.vector_norm(t, dim=-1),
        ):
            self.assertEqual(self._fired_count(lambda: self._raises_or_none(fn, xi)), 0)

    @staticmethod
    def _raises_or_none(fn, x):
        try:
            return fn(x)
        except RuntimeError:
            return None

    def test_noncontiguous_is_served(self):
        # A layout is never a reason to decline: the general arm addresses through the TI
        # offset decode, so a non-contiguous input is SERVED, not handed back.
        xt = torch.randn(64, 128, device="cuda").t()
        for fn in (
            lambda t: torch.sum(t, dim=-1),
            lambda t: torch.sum(t, dim=0),
            lambda t: torch.sum(t),  # reduce-ALL of a non-contiguous input
            lambda t: torch.amax(t, dim=-1),
            lambda t: torch.var(t, dim=-1),
            lambda t: torch.argmax(t, dim=-1),
            # An index over ALL dims is served by materializing first: the index must be a
            # position in the LOGICAL flattening, not in storage order.
            lambda t: torch.argmax(t),
            lambda t: torch.argmin(t),
        ):
            with self.subTest(fn=fn):
                self.assertEqual(self._fired_count(lambda: fn(xt)), 1)
                with _disabled():
                    ref = fn(xt)
                self.assertEqual(fn(xt), ref, exact_dtype=True)

    def test_scalar_is_served(self):
        # A 0-dim input reduces to ITSELF, and aten accepts exactly dim None/[]/0/-1 for it.
        # Both are served: the reduction is over the single element, and the result is 0-dim
        # whatever keepdim says. The dim normalization must not compute `d % ndim` here
        # (ndim is 0) -- for the cond, where it would crash the dispatcher, or for the impl.
        s = torch.tensor(3.5, device="cuda")
        for fn in (
            lambda t: torch.sum(t),
            lambda t: torch.sum(t, dim=0),
            lambda t: torch.sum(t, dim=-1),
            lambda t: torch.sum(t, dim=0, keepdim=True),
            lambda t: torch.amax(t),
            lambda t: torch.argmax(t),
            lambda t: torch.max(t, dim=0),
            lambda t: torch.aminmax(t, dim=0),
            lambda t: torch.count_nonzero(t),
        ):
            with self.subTest(fn=fn):
                self.assertGreaterEqual(self._fired_count(lambda: fn(s)), 1)
                with _disabled():
                    ref = fn(s)
                self.assertEqual(fn(s), ref, exact_dtype=True)
        # An out-of-range or duplicated dim still has to reach aten's error.
        for bad in (1, -2, [0, -1]):
            with self.subTest(dim=bad), self.assertRaises((RuntimeError, IndexError)):
                torch.sum(s, dim=bad)

    def test_invalid_dim_defers_to_aten(self):
        # dim args aten rejects (out-of-range / duplicate) must surface aten's
        # normal error -- the cond declines so aten validates, no wrapped result.
        x = torch.randn(4, 5, 6, device="cuda")
        with self.assertRaises(IndexError):
            torch.sum(x, dim=3)
        with self.assertRaises(RuntimeError):
            torch.sum(x, dim=(0, 0))

    def test_cow_input_served_and_preserved(self):
        # A copy-on-write input is SERVED by our kernel (it exports read-only via
        # ReadOnlyTensorWrapper -> from_dlpack reads through const_data_ptr()), and
        # must stay COW after -- reading it must not materialize it.
        base = torch.randn(128, 512, device="cuda")
        x = torch._lazy_clone(base)
        self.assertEqual(self._fired_count(lambda: torch.sum(x, dim=-1)), 1)
        self.assertTrue(torch._C._is_cow_tensor(x))

    def test_fast_geometry_routing(self):
        # `fast_kind` is a ROUTER, not a gate: a geometry it cannot reshape onto the fast
        # row/column paths is served by the general arm rather than declined. So every case
        # here fires -- what differs is only which path takes it.
        served = [
            ("2D last-dim", torch.randn(512, 512, device="cuda"), -1),
            ("2D dim0", torch.randn(512, 512, device="cuda"), 0),
            (
                "3D last-dim coalesces to row",
                torch.randn(64, 32, 512, device="cuda"),
                -1,
            ),
            (
                "3D dims (1,2) coalesce to row",
                torch.randn(128, 32, 32, device="cuda"),
                (1, 2),
            ),
        ]
        for name, x, dim in served:
            self.assertEqual(
                self._fired_count(lambda: torch.sum(x, dim=dim)),
                1,
                f"{name} should fire",
            )
        # These reach no fast path (mid-dim, transposed, gapped, window-overlapping), so the
        # general arm serves them -- and must get the same answer aten does.
        general = [
            ("3D mid-dim", torch.randn(512, 512, 64, device="cuda"), 1),
            ("transposed", torch.randn(512, 512, device="cuda").t(), -1),
            ("gapped slice", torch.randn(512, 512, device="cuda")[:, ::2], -1),
            (
                "permuted 3D",
                torch.randn(32, 64, 128, device="cuda").permute(2, 0, 1),
                2,
            ),
            (
                "overlapping windows",
                torch.randn(64, 128, device="cuda").unfold(1, 4, 2),
                -1,
            ),
        ]
        for name, x, dim in general:
            self.assertEqual(
                self._fired_count(lambda: torch.sum(x, dim=dim)),
                1,
                f"{name} must be served, not declined",
            )
            with _disabled():
                ref = torch.sum(x, dim=dim)
            self.assertEqual(torch.sum(x, dim=dim), ref, atol=1e-3, rtol=1e-3)

    @unittest.skipUnless(torch.cuda.device_count() >= 2, "needs >= 2 GPUs")
    def test_other_device_defers(self):
        # A tensor not on the current device must fall back (kernel/stream caches
        # are current-device-bound). cuda:1 with cuda:0 current -> aten.
        x = torch.randn(128, 512, device="cuda:1")
        self.assertEqual(self._fired_count(lambda: torch.sum(x, dim=-1)), 0)

    def test_graph_capturable(self):
        # The override must capture into a CUDA graph and replay correctly (the
        # earlier _stream() bug made cute launches deadlock / produce empty graphs).
        x = torch.randn(8192, 1024, device="cuda")
        f = lambda: torch.sum(x, dim=-1)  # noqa: E731
        with _disabled():
            ref = f()
        for _ in range(3):
            f()
        torch.cuda.synchronize()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                f()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out = f()
        g.replay()
        torch.cuda.synchronize()
        self.assertEqual(out, ref, atol=1e-2, rtol=1e-2)

    def test_scalar_operand_is_served(self):
        # aten's unboxed parser turns `x * 2.0` into mul.Tensor(Tensor, Scalar) and
        # dispatches THAT overload -- the number never reaches aten::mul.Scalar. The
        # router now coerces the number to a 0-d tensor BEFORE the cond (it used to do
        # so only on the fallback path), and the cond treats a 0-d CPU operand as that
        # coerced number, so scalar calls are served instead of always declining.
        try:
            from torch._native.ops.pointwise import kernel as K
        except (ModuleNotFoundError, FileNotFoundError):
            # The coercion is router-level, but the only thing that OBSERVES it is a pointwise
            # kernel launch, and that family lands further up the stack. FileNotFoundError as
            # well as ModuleNotFoundError: a leftover __pycache__ makes the editable-install
            # finder claim the package and then fail on the absent source.
            self.skipTest("the pointwise override family is not in this commit")

        def fired(fn):
            # Count real kernel launches -- plan-cache growth is not a liveness signal
            # once the cache is warm from an earlier identical call.
            orig, n = K.run, [0]

            def counting(*a, **k):
                n[0] += 1
                return orig(*a, **k)

            K.run = counting
            try:
                fn()
            finally:
                K.run = orig
            return n[0]

        x = torch.randn(1024, device="cuda")
        for fn in (
            lambda: x * 2.0,
            lambda: x + 2.0,
            lambda: x - 2.0,
            lambda: x / 2.0,
            lambda: torch.fmod(x, 2.0),
        ):
            got = fn()
            with _disabled():
                ref = fn()
            self.assertEqual(got, ref)
            self.assertEqual(fired(fn), 1, "scalar call should route to our kernel")
        # WEAK promotion must survive: a python number never widens the tensor dtype.
        b = torch.randn(64, device="cuda", dtype=torch.bfloat16)
        self.assertEqual((b * 2.0).dtype, torch.bfloat16)
        i = torch.randint(0, 9, (64,), device="cuda", dtype=torch.int32)
        self.assertEqual((i * 2).dtype, torch.int32)  # int stays int
        self.assertEqual((i + 0.5).dtype, torch.float32)  # float promotes the category
        # A dim>0 CPU tensor is a genuine cross-device call, not a scalar -> aten raises.
        with self.assertRaises(RuntimeError):
            x + torch.randn(1024)

    def test_pow_scalar_base_on_cuda(self):
        # REGRESSION: aten's pow_Scalar_out builds a wrapped_scalar_tensor on the
        # EXPONENT's device and redispatches to pow_out (Pow.cpp), so a CUDA
        # wrapped-number tensor reached our Python router; the boxed->Python
        # conversion asserts is_cpu() and this raised INTERNAL ASSERT. The assert
        # fires before any cond runs, so pow's .out overload is left unregistered.
        x = torch.tensor([1.0, 2.0], device="cuda")
        self.assertEqual((2.0**x).cpu(), torch.tensor([2.0, 4.0]))
        self.assertEqual(torch.pow(2.0, x).cpu(), torch.tensor([2.0, 4.0]))
        # float_power always promotes to double
        self.assertEqual(
            torch.float_power(2.0, x).cpu(),
            torch.tensor([2.0, 4.0], dtype=torch.float64),
        )
        xi = torch.tensor([1, 2], dtype=torch.int32, device="cuda")
        self.assertEqual(torch.ldexp(xi, xi).cpu(), torch.tensor([2.0, 8.0]))
        # the tensor-tensor overload we DO serve still works
        self.assertEqual(torch.pow(x, x).cpu(), torch.tensor([1.0, 4.0]))

    def test_sub_warp_row_width_does_not_crash(self):
        # REGRESSION: the tpr ladder's small-N rungs return 8/16 threads per row,
        # and the cross-thread reduce divides by warps_per_row = tpr // WARP, which
        # floored to 0 -> ZeroDivisionError at trace time. That was a hard crash on
        # ordinary calls (x.sum(dim=1) for N=32/64/128), not a fallback. tpr is now
        # floored at one warp. Non-monotonic in N, so cover the whole small range.
        for n in (8, 16, 24, 32, 33, 48, 63, 64, 96, 128, 192):
            x = torch.rand(257, n, device="cuda")
            self.assertEqual(
                torch.sum(x, dim=1), x.double().sum(dim=1).float(), atol=1e-3, rtol=1e-3
            )
            torch.linalg.vector_norm(x, 2, dim=1)  # same reduce path, must not raise

    def test_strided_single_element_view_is_served(self):
        # REGRESSION: is_contiguous() is True for ANY single-element tensor whatever its
        # stride (with one element the stride is unobservable -- every stride addresses
        # the same element), so a.diagonal(offset=2) on (5,3) is a contiguous shape-(1,)
        # tensor that still declares stride (4,). The DSL compares the declared stride
        # against stride_order and rejected it ("The stride_order is not consistent with
        # the layout") -- a hard error on an ordinary sum, not a fallback. The wrap now
        # restrides such a tensor to the canonical form, so these are SERVED (declining
        # would give up coverage for a difference that cannot be observed).
        a = torch.randn(5, 3, device="cuda", dtype=torch.float64)
        d = a.diagonal(offset=2)
        self.assertEqual(d.shape, torch.Size([1]))
        self.assertNotEqual(d.stride(), (1,))  # the leftover stride is the whole point
        for fn in (lambda z: z.sum(), lambda z: z.mean()):
            with _disabled():
                ref = fn(d)
            self.assertEqual(fn(d), ref)

    def test_misaligned_and_lazy_metadata_inputs_decline(self):
        # Two cond gates that cannot be expressed as numerics:
        #   - A base pointer that is not 16-byte aligned: the row/col/xcta wraps claim an
        #     N-derived alignment that from_dlpack VALIDATES, and the compiled kernel
        #     BAKES its load width, so a plan built for an aligned call cannot serve a
        #     misaligned one (clamping the claim per call is not enough). Raised
        #     "Misaligned Tensor data" mid-call on an ordinary sum over a slice.
        #   - A NEG/CONJ bit: lazy metadata, so the exported buffer holds the UNNEGATED
        #     values. It must not be resolved in a cond either, since aten materializes
        #     such a view by CALLING copy_ -- any override of copy_ is then re-entered.
        base = torch.arange(4096, device="cuda", dtype=torch.float64) + 1
        for off in (0, 1, 2, 3):
            t = base[off : off + 512].view(256, 2)
            for fn in (
                lambda z: z.sum(dim=0),
                lambda z: z.sum(dim=1),
                lambda z: z.sum(),
            ):
                with _disabled():
                    ref = fn(t)
                self.assertEqual(fn(t), ref, f"off={off}")
        n = torch.randn(64, device="cuda", dtype=torch.float64)._neg_view()
        self.assertTrue(n.is_neg())
        for fn in (lambda z: z.sum(dim=0), lambda z: z.sum(), lambda z: z.mean()):
            with _disabled():
                ref = fn(n)
            self.assertEqual(fn(n), ref)

    def test_large_scalar_arg_compiles_and_is_not_baked(self):
        # REGRESSION: the DSL mangles every non-IR jit argument's VALUE into the
        # generated MLIR symbol name. Python's repr switches to exponent form at 1e16
        # ("1e+16") and the mangler does not strip "+", so the symbol was unparsable
        # and the compile died with an ICE ("expected '('"). That made any op taking a
        # large scalar fail -- including nan_to_num's DEFAULT posinf, which is the
        # dtype's finite max (3.4e38 for fp32, 1.8e308 for fp64). The kernel takes the
        # scalar as a runtime arg, so it now compiles against a placeholder value.
        from torch._native.ops.pointwise import kernel as K

        x = torch.randn(1024, device="cuda")
        for v in (1e16, 1e20, 3.4e38, -1e20, 1e-20, 2.5):
            got = torch.nn.functional.leaky_relu(x, negative_slope=v)
            with _disabled():
                ref = torch.nn.functional.leaky_relu(x, negative_slope=v)
            self.assertEqual(got, ref)
        # Scalars are runtime args, so many distinct VALUES must share one compile.
        n = len(K._KERNELS)
        for v in (3.0, 4.0, 5.0, 1e17, 1e18):
            torch.nn.functional.leaky_relu(x, negative_slope=v)
        self.assertEqual(len(K._KERNELS), n, "scalar value must not key the kernel")
        # nan_to_num's omitted bounds saturate at the OUTPUT dtype's max, not the fp32
        # compute dtype's (fp16 -> 65504); a wrong fill overflows back to inf.
        vals = [float("nan"), float("inf"), float("-inf"), 1.5]
        sp = torch.tensor(vals, device="cuda")
        for dt in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            t = sp.to(dt)
            with _disabled():
                ref = torch.nan_to_num(t)
            self.assertEqual(torch.nan_to_num(t), ref)

    def test_misaligned_view_shares_no_plan_with_aligned(self):
        # REGRESSION: the plan cache keyed on (shape, stride) but NOT on 16-byte
        # alignment, so base[0:16] and base[1:17].view(16, 2) -- same shape AND stride,
        # different alignment -- shared one plan. The aligned call picks the vec path,
        # which bakes assumed_align=16, so the misaligned call reusing that plan died in
        # from_dlpack with "Misaligned Tensor data on mIns[0]". Alignment is now part of
        # the key, and an out=/in-place target is alignment-gated in the cond too (the
        # kernel compiles against a fresh, always-aligned seed output).
        for dt in (torch.float64, torch.float32, torch.float16, torch.int32):
            base = torch.arange(4096, device="cuda").to(dt) + 1
            for off in (0, 1, 2, 4, 7, 8):
                for shape in ((512,), (256, 2), (64, 8)):
                    n = math.prod(shape)
                    t = base[off : off + n].view(shape)
                    for fn in (lambda z: z * z, lambda z: -z, torch.sigmoid):
                        with _disabled():
                            ref = fn(t)
                        self.assertEqual(fn(t), ref, f"{dt} off={off} shape={shape}")

    def test_empty_copy_destination_falls_back(self):
        # REGRESSION: copy_ BROADCASTS src up to self, so a 1-element source into an
        # EMPTY destination reached the kernel with a perfectly valid source (the
        # conversion gate only rejects an empty SOURCE). aten treats that as a no-op; for
        # us it is a zero-element grid -> "CUDA Error: cudaErrorInvalidConfiguration", or
        # an invalid cute layout when a shape extent is 0. Reached in practice via
        # linalg.matrix_power(torch.empty(0, 2, 2), 0), which fills an identity.
        dst = torch.empty(0, 2, device="cuda", dtype=torch.float64)
        dst.copy_(torch.ones(1, device="cuda", dtype=torch.float64))  # must not raise
        e = torch.empty(0, 2, 2, device="cuda", dtype=torch.float64)
        with _disabled():
            ref = torch.linalg.matrix_power(e, 0)
        self.assertEqual(torch.linalg.matrix_power(e, 0), ref)

    def test_scalar_in_first_operand_slot_is_served(self):
        # REGRESSION: the cond took its DEVICE reference from operand 0, but aten puts the
        # coerced scalar FIRST for the reflected overloads -- rsub.Scalar is
        # `at::sub(wrapped_scalar, self)`, and remainder.Scalar_Tensor / xlogy.Scalar_Self
        # are declared that way. Operand 0 was then a 0-d CPU tensor, failed the
        # is-this-CUDA test, and every such call declined (`1.0 - t` fired nothing). The
        # reference is now the first operand that is not a coerced scalar.
        from torch._native.ops.pointwise import kernel as K

        def served(fn):
            orig, n = K.run, [0]

            def counting(*a, **k):
                n[0] += 1
                return orig(*a, **k)

            K.run = counting
            try:
                out = fn()
            finally:
                K.run = orig
            return n[0], out

        t = torch.tensor([1.0, 2.0, 3.0], device="cuda")
        i = torch.tensor([2, 4, 6], dtype=torch.int32, device="cuda")
        for fn in (
            lambda: 1.0 - t,
            lambda: torch.remainder(2.0, t),
            lambda: torch.xlogy(2.0, t),
            lambda: torch.bitwise_and(3, i),
        ):
            n, got = served(fn)
            self.assertGreaterEqual(n, 1, "scalar-first call must be served")
            with _disabled():
                self.assertEqual(got, fn())
        # A genuine cross-device call must STILL decline and let aten raise, i.e. the
        # loosened reference must not have loosened the device check itself.
        cpu = torch.tensor([1.0, 2.0, 3.0])
        for fn in (lambda: t + cpu, lambda: cpu + t):
            with self.assertRaises(RuntimeError):
                fn()

    def test_int64_reduction_on_the_column_path(self):
        # REGRESSION: a kernel's own _PART_TORCH map lacked Int64 while kernel_general
        # and kernel_xcta had it, so that path was the one that KeyError'd when allocating
        # an int64 stage-1 partial buffer -- integer reductions accumulate in int64.
        # dim=0 is the column path; dim=1 the row path, as a control.
        import cutlass

        from torch._native.ops._cutedsl import traits as T
        from torch._native.ops.reductions import kernel_general as kg

        x = torch.arange(256 * 64, device="cuda", dtype=torch.int64).reshape(256, 64)
        for dim in (0, 1):
            out = kg.reduce_dim(
                T.SumOps(acc=cutlass.Int64), ("sum_i64",), x, {dim}, torch.int64
            )
            ref = x.sum(dim=dim)
            self.assertEqual(out.reshape(ref.shape), ref, f"dim={dim}")

    def test_vector_norm_ord0_and_nansum(self):
        # ord=0 is the NONZERO COUNT, not a |x|**p sum, so it needs CountNonzeroOps rather
        # than NormOps -- the cond used to decline it outright. And NanSumOps existed in
        # the trait library with zero call sites; wiring it also gets nanmean, which aten
        # decomposes into nansum / isnan.logical_not.sum.
        x = torch.tensor([[0.0, 1.0, 0.0, 2.0], [3.0, 0.0, 0.0, 0.0]], device="cuda")
        for ord_ in (0, 1, 2, 3, float("inf"), float("-inf")):
            got = torch.linalg.vector_norm(x, ord_, dim=1)
            with _disabled():
                self.assertEqual(got, torch.linalg.vector_norm(x, ord_, dim=1))
        nan = float("nan")
        y = torch.tensor(
            [[1.0, nan, 3.0], [nan, nan, nan], [4.0, 5.0, 6.0]], device="cuda"
        )
        for fn in (
            lambda: torch.nansum(y, dim=1),
            lambda: torch.nansum(y, dim=0),
            lambda: torch.nansum(y),
            lambda: torch.nansum(y, dim=1, dtype=torch.float64),
            lambda: torch.nanmean(y, dim=1),
        ):
            with _disabled():
                ref = fn()
            self.assertEqual(fn(), ref, equal_nan=True)

    def test_nullary_fill_serves_every_layout(self):
        # fill_ is the NULLARY (nin == 0) case: no input tensor at all, so the caller's
        # `self` is the only source of shape/device/layout AND the destination. Everything
        # in the kernel that normally reads inputs[0] has to come from the output instead.
        #
        # Coverage is the contract -- every layout is SERVED, not just the fast ones: a
        # contiguous aligned target takes the vec path, and unaligned / transposed /
        # strided targets fall to the strided route (which bakes the real layout, hence
        # the output layout appearing in both the plan and kernel keys).
        from torch._native.ops.pointwise import kernel as K

        def served(fn):
            orig, n = K.run, [0]

            def counting(*a, **k):
                n[0] += 1
                return orig(*a, **k)

            K.run = counting
            try:
                fn()
            finally:
                K.run = orig
            return n[0]

        base = torch.empty(4096, device="cuda")
        targets = (
            ("contiguous", torch.empty(1024, device="cuda")),
            ("unaligned", base[1:1025]),
            ("transposed", torch.empty(32, 32, device="cuda").t()),
            ("strided", torch.empty(2048, device="cuda")[::2]),
            ("0-dim", torch.empty((), device="cuda")),
        )
        for name, t in targets:
            self.assertEqual(
                served(lambda t=t: t.fill_(2.5)), 1, f"{name} must be served"
            )
            self.assertTrue(bool((t == 2.5).all()), name)
        # Every constructor aten builds on empty()+fill_ rides the same override, with no
        # per-constructor row: full / zeros / ones / full_like, floats and ints alike.
        for fn in (
            lambda: torch.full((1024,), 3.5, device="cuda"),
            lambda: torch.zeros(1024, device="cuda"),
            lambda: torch.ones(1024, device="cuda"),
            lambda: torch.full((1024,), 9, device="cuda", dtype=torch.int64),
        ):
            self.assertEqual(served(fn), 1)
            with _disabled():
                ref = fn()
            self.assertEqual(fn(), ref)
        # An fp64 value must not narrow through fp32 (the const has to be boxed in the
        # compute dtype: 1e20 came back as its fp32 neighbour before that fix).
        d = torch.empty(8, device="cuda", dtype=torch.float64).fill_(1e20)
        self.assertEqual(d[0].item(), 1e20)

    def test_range_factories_use_the_flat_index(self):
        # arange/linspace are nullary AND index-consuming: each element's value comes from
        # its FLAT INDEX alone, which aten expresses with gpu_kernel_with_index and we
        # expose as the kernel's `with_index` flag (strided route only -- the vectorized
        # routes hand the fn a whole V-wide fragment, so there is no single index).
        from torch._native.ops.pointwise import kernel as K

        def served(fn):
            orig, n = K.run, [0]

            def counting(*a, **k):
                n[0] += 1
                return orig(*a, **k)

            K.run = counting
            try:
                out = fn()
            finally:
                K.run = orig
            return n[0], out

        # linspace: fp32/fp64 and the integer dtypes are BIT-EXACT with aten.
        for dt in (torch.float32, torch.float64, torch.int32, torch.int64):
            for a, b, steps in ((0, 1, 5), (-1, 1, 9), (5, -5, 64), (2.5, 3.5, 17)):
                fn = lambda: torch.linspace(  # noqa: E731
                    a, b, steps, device="cuda", dtype=dt
                )
                n, got = served(fn)
                self.assertEqual(n, 1, f"linspace {dt} must be served")
                with _disabled():
                    self.assertEqual(got, fn())
        # Halves compute in FP32 and narrow only on the store, where aten runs the whole
        # expression in scalar_t -- so we can differ by well under one ULP, on the MORE
        # accurate side. Endpoints stay exact regardless (that is what aten's halfway
        # split buys, and why the kernel reproduces it rather than stepping forward
        # throughout).
        for dt in (torch.float16, torch.bfloat16):
            for a, b, steps in ((0, 1, 5), (-1, 1, 64), (5, -5, 1001)):
                got = torch.linspace(a, b, steps, device="cuda", dtype=dt)
                with _disabled():
                    ref = torch.linspace(a, b, steps, device="cuda", dtype=dt)
                span = max(abs(float(a)), abs(float(b)))
                tol = torch.finfo(dt).eps * span
                self.assertLess((got.double() - ref.double()).abs().max().item(), tol)
                self.assertEqual(got[0].item(), torch.tensor(a, dtype=dt).item())
                self.assertEqual(got[-1].item(), torch.tensor(b, dtype=dt).item())
        # arange: we override arange.start_out, which the functional form reaches only
        # from C++ (at::arange_out), so drive the .out form the override actually serves.
        for dt in (torch.float32, torch.float64, torch.int32, torch.int64):
            for s, e, st in ((0, 10, 1), (0.0, 5.0, 0.5), (10, 0, -1), (-5, 5, 2)):
                with _disabled():
                    ref = torch.arange(s, e, st, device="cuda", dtype=dt)
                out = torch.empty_like(ref)
                n, got = served(lambda out=out: torch.arange(s, e, st, out=out))
                self.assertEqual(n, 1, f"arange {dt} must be served")
                self.assertEqual(got, ref)

    def test_optional_scalar_is_not_silently_dropped(self):
        # REGRESSION (wrong RESULTS, not a crash): logit's eps is `float?`, and the row
        # originally declared no scalars at all on the theory that an explicit eps would
        # decline. It did not -- the arg was silently IGNORED, so torch.logit(x, 1e-3)
        # ran the unclamped kernel and returned nan where aten clamps. An omitted eps
        # means "no clamping", which aten spells as a negative sentinel; optional_defaults
        # supplies it, so one row serves both overloads.
        x = torch.tensor(
            [0.0, 1e-8, 0.5, 0.9999, 1.5, -0.5, float("nan")], device="cuda"
        )
        for fn in (
            lambda t: torch.logit(t),
            lambda t: torch.logit(t, 1e-3),
            lambda t: torch.logit(t, 0.0),
            lambda t: torch.special.logit(t, eps=0.1),
            # eps > 0.5 CROSSES the bounds (lo=0.6 > hi=0.4). aten's nested ternary
            # returns lo and never re-clamps; a sequential clamp-low-then-high would pull
            # it back to hi and FLIP THE SIGN of the log (caught by test_out_logit).
            lambda t: torch.logit(t, 0.6),
        ):
            with _disabled():
                ref = fn(x)
            self.assertEqual(fn(x), ref, equal_nan=True)

    def test_pointwise_neg_and_conj_views_decline(self):
        # REGRESSION: a neg/conj bit is LAZY metadata -- the buffer holds the UNNEGATED
        # values -- and aten materializes such a view BY CALLING copy_, which THIS
        # commit's copy_ override intercepts. Resolving it inside a cond therefore
        # recursed until the stack blew: a plain torch.sin on a _neg_view() input raised
        # RecursionError (also relu, mul, clone, and every reduction, since they all
        # funnel through the same copy_). Declining lets aten resolve the bit.
        x = torch.randn(64, device="cuda", dtype=torch.float64)
        n = x._neg_view()
        self.assertTrue(n.is_neg())
        for fn in (
            torch.sin,
            torch.relu,
            torch.nan_to_num,
            lambda t: t * 2.0,
            lambda t: torch.clamp(t, -1.0, 1.0),
            lambda t: t.clone(),
            lambda t: t.float(),
        ):
            with _disabled():
                ref = fn(n)
            self.assertEqual(fn(n), ref)  # must not raise RecursionError
        # A conj view of a complex tensor is likewise declined rather than misread.
        c = torch.randn(64, device="cuda", dtype=torch.complex64).conj()
        with _disabled():
            ref = torch.real(c)
        self.assertEqual(torch.real(c), ref)


if __name__ == "__main__":
    run_tests()
