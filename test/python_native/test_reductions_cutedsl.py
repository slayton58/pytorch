# Owner(s): ["module: dsl-native-ops"]
#
# Wiring tests for routing, fallback, and CUDA graph capture. OpInfo tests cover
# numerical behavior.

import math
import sys
import unittest
import warnings

import torch
from torch.testing._internal.common_cuda import TEST_CUDA
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TEST_CUTEDSL,
    TestCase,
)


if not TEST_CUTEDSL:
    sys.stderr.write("CuTeDSL not available\n")
    if __name__ == "__main__":
        sys.exit(0)
    raise unittest.SkipTest("CuTeDSL not available")

from torch._native.ops.reductions import (
    kernel_coltile,
    kernel_general,
    kernel_rowtile,
    kernel_xcta,
)


def _disabled():
    return torch.backends.python_native.cutedsl.disabled()


_OUT_CASES = (
    ("sum_dim", lambda x, o: torch.sum(x, dim=1, out=o[0]), ((8,),), (torch.float32,)),
    (
        "sum_full",
        lambda x, o: torch.ops.aten.sum.out(x, out=o[0]),
        ((),),
        (torch.float32,),
    ),
    (
        "mean_dim",
        lambda x, o: torch.mean(x, dim=1, out=o[0]),
        ((8,),),
        (torch.float32,),
    ),
    ("mean_full", lambda x, o: torch.mean(x, out=o[0]), ((),), (torch.float32,)),
    (
        "nansum",
        lambda x, o: torch.nansum(x, dim=1, out=o[0]),
        ((8,),),
        (torch.float32,),
    ),
    ("amax", lambda x, o: torch.amax(x, dim=1, out=o[0]), ((8,),), (torch.float32,)),
    ("amin", lambda x, o: torch.amin(x, dim=1, out=o[0]), ((8,),), (torch.float32,)),
    (
        "prod_dim",
        lambda x, o: torch.prod(x, dim=1, out=o[0]),
        ((8,),),
        (torch.float32,),
    ),
    (
        "prod_full",
        lambda x, o: torch.ops.aten.prod.out(x, out=o[0]),
        ((),),
        (torch.float32,),
    ),
    (
        "argmax",
        lambda x, o: torch.argmax(x, dim=1, out=o[0]),
        ((8,),),
        (torch.int64,),
    ),
    (
        "argmin",
        lambda x, o: torch.argmin(x, dim=1, out=o[0]),
        ((8,),),
        (torch.int64,),
    ),
    (
        "max_dim",
        lambda x, o: torch.max(x, dim=1, out=o),
        ((8,), (8,)),
        (torch.float32, torch.int64),
    ),
    ("max_full", lambda x, o: torch.max(x, out=o[0]), ((),), (torch.float32,)),
    (
        "min_dim",
        lambda x, o: torch.min(x, dim=1, out=o),
        ((8,), (8,)),
        (torch.float32, torch.int64),
    ),
    ("min_full", lambda x, o: torch.min(x, out=o[0]), ((),), (torch.float32,)),
    ("var", lambda x, o: torch.var(x, dim=1, out=o[0]), ((8,),), (torch.float32,)),
    ("std", lambda x, o: torch.std(x, dim=1, out=o[0]), ((8,),), (torch.float32,)),
    (
        "vector_norm",
        lambda x, o: torch.linalg.vector_norm(x, dim=1, out=o[0]),
        ((8,),),
        (torch.float32,),
    ),
    ("all_dim", lambda x, o: torch.all(x, dim=1, out=o[0]), ((8,),), (torch.uint8,)),
    (
        "all_dims",
        lambda x, o: torch.all(x, dim=(0, 1), out=o[0]),
        ((),),
        (torch.uint8,),
    ),
    ("all_full", lambda x, o: torch.all(x, out=o[0]), ((),), (torch.uint8,)),
    ("any_dim", lambda x, o: torch.any(x, dim=1, out=o[0]), ((8,),), (torch.uint8,)),
    (
        "any_dims",
        lambda x, o: torch.any(x, dim=(0, 1), out=o[0]),
        ((),),
        (torch.uint8,),
    ),
    ("any_full", lambda x, o: torch.any(x, out=o[0]), ((),), (torch.uint8,)),
    (
        "count_nonzero_dim",
        lambda x, o: torch.ops.aten.count_nonzero.out(x, 1, out=o[0]),
        ((8,),),
        (torch.int64,),
    ),
    (
        "count_nonzero_dims",
        lambda x, o: torch.ops.aten.count_nonzero.dim_IntList_out(x, [0, 1], out=o[0]),
        ((),),
        (torch.int64,),
    ),
    (
        "var_mean",
        lambda x, o: torch.ops.aten.var_mean.correction_out(
            x, [1], correction=1, keepdim=False, out0=o[0], out1=o[1]
        ),
        ((8,), (8,)),
        (torch.float32, torch.float32),
    ),
    (
        "std_mean",
        lambda x, o: torch.ops.aten.std_mean.correction_out(
            x, [1], correction=1, keepdim=False, out0=o[0], out1=o[1]
        ),
        ((8,), (8,)),
        (torch.float32, torch.float32),
    ),
    (
        "aminmax",
        lambda x, o: torch.aminmax(x, dim=1, out=o),
        ((8,), (8,)),
        (torch.float32, torch.float32),
    ),
)


@unittest.skipUnless(TEST_CUDA, "CUDA required")
class TestCuTeDSLReductionWiring(TestCase):
    def _fired_count(self, fn):
        names = ("reduce_dim", "reduce_dim2", "reduce_all", "reduce_all2")
        orig = {nm: getattr(kernel_general, nm) for nm in names}
        n = [0]

        def wrap(f):
            def counting(*a, **k):
                n[0] += 1
                return f(*a, **k)

            return counting

        for nm in names:
            setattr(kernel_general, nm, wrap(orig[nm]))
        try:
            fn()
        finally:
            for nm in names:
                setattr(kernel_general, nm, orig[nm])
        return n[0]

    def test_unsupported_dtype_falls_back(self):
        for dtype in (torch.float64, torch.int64, torch.complex64, torch.complex128):
            xi = torch.ones(64, 64, device="cuda", dtype=dtype)
            with self.subTest(dtype=dtype):
                self.assertEqual(self._fired_count(lambda: torch.sum(xi, dim=-1)), 0)
                with _disabled():
                    ref = torch.sum(xi, dim=-1)
                self.assertEqual(torch.sum(xi, dim=-1), ref)

    def test_argmax_declines_bool(self):
        x = torch.ones(64, 64, device="cuda", dtype=torch.bool)
        with self.assertRaisesRegex(RuntimeError, "does not support bool"):
            torch.argmax(x, dim=-1)

    def test_noncontiguous_full_reductions_are_served(self):
        xt = torch.randn(64, 128, device="cuda").t()
        for fn in (
            lambda t: torch.sum(t),
            lambda t: torch.argmax(t),
        ):
            with self.subTest(fn=fn):
                self.assertEqual(self._fired_count(lambda: fn(xt)), 1)
                with _disabled():
                    ref = fn(xt)
                self.assertEqual(fn(xt), ref, exact_dtype=True)

    def test_empty_reduction_without_identity_raises(self):
        empty_axis = torch.randn(5, 0, device="cuda")
        for fn in (torch.amax, torch.amin, torch.argmax, torch.aminmax):
            with self.subTest(fn=fn):
                with self.assertRaises((RuntimeError, IndexError)):
                    fn(empty_axis, dim=1)

    @staticmethod
    def _compiled_kernel_count():
        caches = (
            kernel_general._COMPILE_CACHE,
            kernel_rowtile._CACHE,
            kernel_coltile._CACHE,
            kernel_xcta._PLAN,
        )
        return sum(len(c) for c in caches), caches

    def test_kernel_count_does_not_scale_with_shape(self):
        # These sizes share a vector class and bucket, so they must share kernels.
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

    def test_scalar_is_served(self):
        s = torch.tensor(3.5, device="cuda")
        for fn in (
            lambda t: torch.sum(t),
            lambda t: torch.sum(t, dim=-1),
            lambda t: torch.sum(t, dim=0, keepdim=True),
        ):
            with self.subTest(fn=fn):
                self.assertGreaterEqual(self._fired_count(lambda: fn(s)), 1)
                with _disabled():
                    ref = fn(s)
                self.assertEqual(fn(s), ref, exact_dtype=True)
        for bad in (1, -2, [0, -1]):
            with self.subTest(dim=bad), self.assertRaises((RuntimeError, IndexError)):
                torch.sum(s, dim=bad)

    def test_cow_input_served_and_preserved(self):
        # A COW input is SERVED (it exports read-only, so from_dlpack reads const_data_ptr()) and
        # must STAY COW: reading it must not materialize it.
        base = torch.randn(128, 512, device="cuda")
        x = torch._lazy_clone(base)
        self.assertEqual(self._fired_count(lambda: torch.sum(x, dim=-1)), 1)
        self.assertTrue(torch._C._is_cow_tensor(x))

    def test_fast_geometry_routing(self):
        served = [
            ("2D last-dim", torch.randn(512, 512, device="cuda"), -1),
            ("2D dim0", torch.randn(512, 512, device="cuda"), 0),
            (
                "3D last-dim coalesces to row",
                torch.randn(64, 32, 512, device="cuda"),
                -1,
            ),
        ]
        for name, x, dim in served:
            self.assertEqual(
                self._fired_count(lambda: torch.sum(x, dim=dim)),
                1,
                f"{name} should fire",
            )
        general = [
            ("3D mid-dim", torch.randn(512, 512, 64, device="cuda"), 1),
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
            torch.testing.assert_close(torch.sum(x, dim=dim), ref, atol=1e-3, rtol=1e-3)

    @unittest.skipUnless(torch.cuda.device_count() >= 2, "needs >= 2 GPUs")
    def test_other_device_is_served(self):
        x0 = torch.randn(128, 512, device="cuda:0")
        x1 = x0.to("cuda:1")
        with _disabled():
            expected = torch.sum(x1, dim=-1)
        with torch.cuda.device(0):
            torch.sum(x0, dim=-1)
            self.assertEqual(self._fired_count(lambda: torch.sum(x1, dim=-1)), 1)
            actual = torch.sum(x1, dim=-1)
            self.assertEqual(torch.cuda.current_device(), 0)
        torch.testing.assert_close(actual, expected, atol=1e-3, rtol=1e-3)

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
        torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)

    def test_sub_warp_row_width_does_not_crash(self):
        # REGRESSION: the small-N rungs returned 8/16 threads per row and warps_per_row floored to 0,
        # a trace-time ZeroDivisionError on an ordinary sum. Non-monotonic in N, so cover the range.
        for n in (8, 16, 33, 64, 96, 192):
            x = torch.rand(257, n, device="cuda")
            torch.testing.assert_close(
                torch.sum(x, dim=1), x.double().sum(dim=1).float(), atol=1e-3, rtol=1e-3
            )
            torch.linalg.vector_norm(x, 2, dim=1)  # same reduce path, must not raise

    def test_strided_single_element_view_is_served(self):
        # REGRESSION: is_contiguous() is True for ANY single-element tensor, since with one element
        # the stride is unobservable -- so a.diagonal(offset=2) is a contiguous shape-(1,) tensor that
        # still declares stride (4,), which the DSL rejected outright. The wrap restrides it, and
        # these are SERVED: declining would give up coverage for a difference nothing can observe.
        a = torch.randn(5, 3, device="cuda")
        d = a.diagonal(offset=2)
        self.assertEqual(d.shape, torch.Size([1]))
        self.assertNotEqual(d.stride(), (1,))  # the leftover stride is the whole point
        with _disabled():
            ref = d.sum()
        self.assertEqual(d.sum(), ref)

    def test_misaligned_inputs_are_served(self):
        cases = (
            ("row", (256, 2), lambda t: torch.amax(t, dim=1)),
            ("cross-CTA row", (2, 32768), lambda t: torch.amax(t, dim=1)),
            ("column", (4096, 256), lambda t: torch.amax(t, dim=0)),
            ("general", (8, 16, 32), lambda t: torch.sum(t, dim=1)),
        )
        for name, shape, fn in cases:
            base = torch.randn(math.prod(shape) + 1, device="cuda")
            t = base[1:].view(shape)
            self.assertNotEqual(t.data_ptr() % 16, 0)
            with _disabled():
                expected = fn(t)
            with self.subTest(name=name):
                self.assertEqual(self._fired_count(lambda: fn(t)), 1)
                torch.testing.assert_close(fn(t), expected, atol=1e-3, rtol=1e-3)

    def test_lazy_negative_input_is_served(self):
        n = torch.randn(64, device="cuda")._neg_view()
        self.assertTrue(n.is_neg())
        with _disabled():
            ref = n.sum()
        self.assertEqual(self._fired_count(lambda: n.sum()), 1)
        self.assertEqual(n.sum(), ref)

    def test_invalid_degrees_of_freedom_warn_and_are_served(self):
        x = torch.randn(4, 16, device="cuda")
        cases = (
            ("var", lambda: torch.var(x, dim=1, correction=16)),
            ("std", lambda: torch.std(x, dim=1, correction=16)),
            ("var_mean", lambda: torch.var_mean(x, dim=1, correction=16)),
            ("std_mean", lambda: torch.std_mean(x, dim=1, correction=16)),
        )
        for name, fn in cases:
            with (
                _disabled(),
                self.assertWarnsRegex(UserWarning, "degrees of freedom is <= 0"),
            ):
                expected = fn()
            actual = []
            with (
                self.subTest(name=name),
                self.assertWarnsRegex(UserWarning, "degrees of freedom is <= 0"),
            ):
                fired = self._fired_count(lambda: actual.append(fn()))
            self.assertEqual(fired, 1)
            self.assertEqual(actual[0], expected)

    @parametrize(
        "correction",
        [float("-inf"), float("inf"), float("nan")],
        name_fn=lambda correction: str(correction),
    )
    def test_nonfinite_correction_is_served(self, correction):
        x = torch.tensor([[1.0, 2.0], [1.0, 1.0]], device="cuda")
        with _disabled(), warnings.catch_warnings(record=True) as expected_warnings:
            expected = torch.var_mean(x, dim=1, correction=correction)
        actual = []
        with warnings.catch_warnings(record=True) as actual_warnings:
            fired = self._fired_count(
                lambda: actual.append(torch.var_mean(x, dim=1, correction=correction))
            )
        self.assertEqual(fired, 1)
        self.assertEqual(actual[0], expected)
        self.assertEqual(
            ["degrees of freedom is <= 0" in str(w.message) for w in actual_warnings],
            ["degrees of freedom is <= 0" in str(w.message) for w in expected_warnings],
        )


@unittest.skipUnless(TEST_CUDA, "CUDA required")
class TestCuTeDSLReductionOut(TestCase):
    @parametrize(
        "name,call,shapes,dtypes",
        _OUT_CASES,
        name_fn=lambda name, call, shapes, dtypes: name,
    )
    def test_out_overload(self, device, name, call, shapes, dtypes):
        x = torch.randn(8, 16, device=device)
        if name.startswith("nansum"):
            x[0, 0] = float("nan")

        def make_outs():
            return tuple(
                torch.empty(shape, device=device, dtype=dtype)
                for shape, dtype in zip(shapes, dtypes)
            )

        expected = make_outs()
        with _disabled():
            call(x, expected)

        names = ("reduce_dim", "reduce_dim2", "reduce_all")
        original = {entry: getattr(kernel_general, entry) for entry in names}
        fired = [0]

        def wrap(fn):
            def counting(*args, **kwargs):
                fired[0] += 1
                return fn(*args, **kwargs)

            return counting

        for entry, fn in original.items():
            setattr(kernel_general, entry, wrap(fn))
        actual = make_outs()
        if name == "sum_dim":
            actual = (torch.empty(1, device=device, dtype=dtypes[0]),)
        try:
            if name == "sum_dim":
                with self.assertWarnsRegex(
                    UserWarning, "An output with one or more elements was resized"
                ):
                    result = call(x, actual)
            else:
                result = call(x, actual)
        finally:
            for entry, fn in original.items():
                setattr(kernel_general, entry, fn)

        self.assertGreater(fired[0], 0)
        returned = result if isinstance(result, tuple) else (result,)
        for got, out in zip(returned, actual):
            self.assertIs(got, out)
        for got, ref in zip(actual, expected):
            tol = 1e-10 if ref.dtype is torch.float64 else 1e-2
            torch.testing.assert_close(got, ref, rtol=tol, atol=tol)


instantiate_parametrized_tests(TestCuTeDSLReductionWiring)
instantiate_device_type_tests(TestCuTeDSLReductionOut, globals(), only_for="cuda")


if __name__ == "__main__":
    run_tests()
