# Owner(s): ["module: dsl-native-ops"]

import os
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

import torch
from torch.testing._internal.common_cuda import SM90OrLater, TEST_CUDA
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

from torch._native.ops.reductions import kernel_general as kg, kernel_rowtile as rt


class TestReductionConfig(TestCase):
    def general(self, **updates):
        args = dict(
            cc=(10, 7),
            dtype=torch.float32,
            trait_key="sum",
            count=1024,
            num_o=65536,
            red_pairs=((1024, 2),),
            kept_pairs=((65536, 2048),),
            order="unordered",
            nfields=1,
            nouts=1,
        )
        args.update(updates)
        return kg.select_general_config(**args)

    @parametrize("sum_gate", ["0", "1"])
    @parametrize("native_gate", ["0", "1"])
    def test_explicit_order_and_live_environment(self, sum_gate, native_gate):
        with mock.patch.dict(
            os.environ,
            {rt._SUM_INNER_TREE_ENV: sum_gate, rt._INNER_TREE_ENV: native_gate},
        ):
            expected = "inner_tree" if "1" in (sum_gate, native_gate) else "unordered"
            self.assertEqual(rt.reduction_order(), expected)
            self.assertEqual(rt.reduction_order("unordered"), "unordered")
            self.assertEqual(rt.reduction_order("inner_tree"), "inner_tree")
        with self.assertRaisesRegex(ValueError, "order"):
            rt.reduction_order("linear")

    @parametrize("dtype", [torch.float32, torch.bfloat16])
    @parametrize("mib", [16, 64, 256, 2048])
    @parametrize("trait_key,nouts", [("var0", 1), ("varmean0", 2)])
    def test_measured_welford_rows(self, dtype, mib, trait_key, nouts):
        itemsize = torch.empty((), dtype=dtype).element_size()
        rows = mib * 2**20 // (1024 * itemsize)
        for order in ("unordered", "inner_tree"):
            with self.subTest(order=order):
                self.assertEqual(
                    rt.select_row_order(
                        (10, 7),
                        dtype,
                        trait_key,
                        1024,
                        rows,
                        order=order,
                        nfields=3,
                        nouts=nouts,
                    ),
                    "inner_tree",
                )
                config = self.general(
                    dtype=dtype,
                    trait_key=trait_key,
                    num_o=rows,
                    kept_pairs=((rows, 2048),),
                    order=order,
                    nfields=3,
                    nouts=nouts,
                )
                self.assertEqual(config.kernel_order, "inner_tree")

    @parametrize("dtype", [torch.float32, torch.bfloat16])
    @parametrize("mib", [256, 2048])
    def test_measured_strided_sum(self, dtype, mib):
        itemsize = torch.empty((), dtype=dtype).element_size()
        rows = mib * 2**20 // (1024 * itemsize)
        config = self.general(dtype=dtype, num_o=rows, kept_pairs=((rows, 2048),))
        self.assertEqual((config.block, config.kernel_order), (64, "linear"))
        ordered = self.general(
            dtype=dtype,
            num_o=rows,
            kept_pairs=((rows, 2048),),
            order="inner_tree",
        )
        self.assertEqual(ordered.kernel_order, "inner_tree")

    @parametrize("cc", [(8, 0), (9, 0), (10, 0), (10, 3), (11, 0)])
    def test_architecture_isolation(self, cc):
        self.assertEqual(self.general(cc=cc), kg._GeneralConfig())
        self.assertEqual(
            rt.select_row_order(
                cc,
                torch.float32,
                "var0",
                1024,
                65536,
                order="unordered",
                nfields=3,
                nouts=1,
            ),
            "linear",
        )
        self.assertEqual(
            self.general(cc=cc, order="inner_tree").kernel_order, "inner_tree"
        )

    @parametrize(
        "updates",
        [
            {"dtype": torch.float16},
            {"dtype": torch.float64},
            {"trait_key": "prod"},
            {"count": 1023, "red_pairs": ((1023, 2),)},
            {"count": 1025, "red_pairs": ((1025, 2),)},
            {"num_o": 65535, "kept_pairs": ((65535, 2048),)},
            {"num_o": 65537, "kept_pairs": ((65537, 2048),)},
            {"num_o": 131072, "kept_pairs": ((131072, 2048),)},
            {"red_pairs": ((1024, 3),)},
            {"kept_pairs": ((65536, 2049),)},
            {"red_pairs": ((32, 2), (32, 64))},
            {"nfields": 3},
            {"nouts": 2},
            {"acc_bits": 64},
            {"alignment": 8},
        ],
    )
    def test_unmeasured_general_inputs_keep_defaults(self, updates):
        self.assertEqual(self.general(**updates), kg._GeneralConfig())

    @parametrize("rows", [65535, 65537, 131072, 524287, 524289])
    def test_row_size_boundaries_keep_defaults(self, rows):
        self.assertEqual(
            rt.select_row_order(
                (10, 7),
                torch.float32,
                "var0",
                1024,
                rows,
                order="unordered",
                nfields=3,
                nouts=1,
            ),
            "linear",
        )

    @parametrize(
        "updates",
        [
            {"trait_key": "var1"},
            {"trait_key": "std0"},
            {"trait_key": "varmean0", "nouts": 1},
            {"dtype": torch.float16},
            {"nfields": 2},
            {"acc_bits": 64},
            {"alignment": 8},
        ],
    )
    def test_unmeasured_row_traits_keep_defaults(self, updates):
        args = dict(
            cc=(10, 7),
            dtype=torch.float32,
            trait_key="var0",
            N=1024,
            M=65536,
            order="unordered",
            nfields=3,
            nouts=1,
        )
        args.update(updates)
        self.assertEqual(rt.select_row_order(**args), "linear")

    @parametrize("mib", [16, 64, 256, 2048])
    def test_measured_bfloat16_row_argmax(self, mib):
        rows = mib * 2**20 // 2048
        self.assertEqual(
            rt.select_row_order(
                (10, 7),
                torch.bfloat16,
                "argmaxi32",
                1024,
                rows,
                order="unordered",
                nfields=2,
                nouts=1,
            ),
            "inner_tree",
        )
        self.assertEqual(
            rt.select_row_order(
                (10, 7),
                torch.float32,
                "argmaxi32",
                1024,
                rows // 2,
                order="unordered",
                nfields=2,
                nouts=1,
            ),
            "linear",
        )

    @parametrize("dtype", [torch.float32, torch.bfloat16])
    @parametrize("mib", [16, 64, 256, 2048])
    @parametrize("trait_key", ["sum", "mean", "amax", "vnorm2", "argmaxi32"])
    def test_measured_strided_simple_ops(self, dtype, mib, trait_key):
        itemsize = torch.empty((), dtype=dtype).element_size()
        rows = mib * 2**20 // (1024 * itemsize)
        config = self.general(
            dtype=dtype,
            trait_key=trait_key,
            num_o=rows,
            kept_pairs=((rows, 2048),),
            nfields=2 if trait_key == "argmaxi32" else 1,
        )
        if trait_key == "sum" and mib in (256, 2048):
            expected = (64, "linear")
        elif trait_key == "argmaxi32":
            inner = mib == 2048 or (dtype == torch.bfloat16 and mib in (64, 256))
            expected = (128, "inner_tree" if inner else "linear")
        else:
            expected = (128, "inner_tree" if mib != 16 else "linear")
        self.assertEqual((config.block, config.kernel_order), expected)

    def test_inner_tree_cannot_decline_into_unordered_row(self):
        x = SimpleNamespace(
            dim=lambda: 2,
            is_cuda=True,
            stride=lambda axis: 1,
            shape=(8, 1024),
            dtype=torch.float32,
            element_size=lambda: 4,
            device="cuda",
        )
        with mock.patch.object(rt, "trait_itree_plan", return_value=None):
            with self.assertRaisesRegex(ValueError, "inner_tree.*cannot"):
                rt.reduce_row_tile(
                    object(), "sum", x, [torch.float32], order="inner_tree"
                )

    def test_rubin_staging_is_explicit_and_unchanged(self):
        self.assertIn((10, 7), rt._ITREE_ARCH)
        self.assertEqual(rt._ITREE_ARCH[(10, 7)], rt._ITREE_ARCH["default"])
        self.assertNotEqual(rt._ITREE_ARCH[(10, 7)], rt._ITREE_ARCH[(10, 0)])

    @parametrize(
        "dtype,width,mib,key,fields,nouts,expected",
        [
            (torch.float32, 256, 16, "mean", 1, 1, "linear"),
            (torch.float32, 256, 64, "mean", 1, 1, "inner_tree"),
            (torch.bfloat16, 256, 16, "sum", 1, 1, "inner_tree"),
            (torch.bfloat16, 256, 64, "sum", 1, 1, "linear"),
            (torch.float32, 4096, 16, "varmean0", 3, 2, "inner_tree"),
            (torch.float32, 4096, 256, "sum", 1, 1, "inner_tree"),
            (torch.bfloat16, 4096, 256, "sum", 1, 1, "linear"),
            (torch.bfloat16, 4096, 2048, "argmaxi32", 2, 1, "inner_tree"),
        ],
    )
    def test_unordered_row_selection(
        self, dtype, width, mib, key, fields, nouts, expected
    ):
        rows = (mib << 20) // (width * dtype.itemsize)
        args = ((10, 7), dtype, key, width, rows)
        kwargs = dict(nfields=fields, nouts=nouts)
        self.assertEqual(
            rt.select_row_order(*args, order="unordered", **kwargs), expected
        )
        self.assertEqual(
            rt.select_row_order(*args, order="inner_tree", **kwargs), "inner_tree"
        )

    @parametrize(
        "updates",
        [
            {"cc": (8, 0)},
            {"cc": (9, 0)},
            {"cc": (10, 0)},
            {"cc": (11, 0)},
            {"N": 4095},
            {"N": 4097},
            {"M": 16383},
            {"M": 16385},
            {"nfields": 2},
            {"nouts": 2},
            {"alignment": 8},
        ],
    )
    def test_unordered_row_selection_guards(self, updates):
        args = dict(
            cc=(10, 7),
            dtype=torch.float32,
            trait_key="var0",
            N=4096,
            M=16384,
            order="unordered",
            nfields=3,
            nouts=1,
        )
        args.update(updates)
        self.assertEqual(rt.select_row_order(**args), "linear")


@unittest.skipUnless(TEST_CUDA and SM90OrLater, "CuTeDSL requires Hopper or later")
class TestReductionConfigDevice(TestCase):
    @parametrize("dtype", [torch.float32, torch.bfloat16])
    @parametrize("stride", [1, 2])
    @parametrize("op", ["sum", "var", "var_mean"])
    @parametrize("pattern", ["signed", "constant", "offset"])
    def test_public_dispatch_mode_changes_and_graphs(
        self, device, dtype, stride, op, pattern
    ):
        from torch import _native as native

        storage = torch.empty((1024, 1024 * stride), device=device, dtype=dtype)
        x = storage[:, ::stride]
        if pattern == "constant":
            x.fill_(3)
        else:
            x.normal_()
            if pattern == "offset":
                x.mul_(8).add_(1024)
        kwargs = {"dim": 1}
        if op == "sum":
            kwargs["dtype"] = torch.float32
        else:
            kwargs["correction"] = 0
        original_input = x.clone()
        rtol = 0.02 if dtype == torch.bfloat16 else 1e-4
        dsl = torch.backends.python_native.cutedsl
        fn = getattr(torch, op)
        select_row = rt.select_row_order
        select_general = kg.select_general_config

        # Exercise the measured rule with small allocations. The kernel still
        # receives the real tensor shape; policy boundaries are tested above.
        def measured_row(cc, dtype, trait_key, N, M, **kw):
            rows = (256 << 20) // (1024 * x.element_size())
            return select_row((10, 7), dtype, trait_key, N, rows, **kw)

        def measured_general(
            cc, dtype, trait_key, count, num_o, red_pairs, kept_pairs, **kw
        ):
            rows = (256 << 20) // (1024 * x.element_size())
            return select_general(
                (10, 7),
                dtype,
                trait_key,
                count,
                rows,
                red_pairs,
                ((rows, 2048),) if stride == 2 else kept_pairs,
                **kw,
            )

        with native._unconditional_masked(), dsl.disabled():
            reference = fn(x, **kwargs)
        old_aot = native.aot_enabled()
        old_enabled = dsl.enabled
        native.set_aot_enabled(False)
        dsl.enable()
        gates = {rt._SUM_INNER_TREE_ENV: "0", rt._INNER_TREE_ENV: "0"}
        try:
            with (
                torch.inference_mode(),
                mock.patch.dict(os.environ, gates),
                mock.patch.object(
                    rt, "select_row_order", side_effect=measured_row
                ) as row_select,
                mock.patch.object(
                    kg, "select_general_config", side_effect=measured_general
                ) as general_select,
                mock.patch.object(kg, "_launch", wraps=kg._launch) as launches,
                mock.patch.object(rt, "_run_itree", wraps=rt._run_itree) as row_tree,
                mock.patch.object(
                    kg, "_try_indexed_itree", wraps=kg._try_indexed_itree
                ) as general_tree,
            ):
                results = []
                for gate in ("0", "1", "0", "1"):
                    x.copy_(original_input)
                    os.environ[rt._INNER_TREE_ENV] = gate
                    launches.reset_mock()
                    row_tree.reset_mock()
                    general_tree.reset_mock()
                    stream = torch.cuda.Stream(device=device)
                    stream.wait_stream(torch.cuda.current_stream(device))
                    with torch.cuda.stream(stream):
                        fn(x, **kwargs)
                    stream.synchronize()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=stream):
                        got = fn(x, **kwargs)
                    graph.replay()
                    torch.cuda.synchronize(device)
                    self.assertEqual(got, reference, atol=1e-4, rtol=rtol)
                    if gate == "1" or op != "sum":
                        self.assertTrue(
                            row_tree.called if stride == 1 else general_tree.called
                        )
                    elif stride == 2:
                        self.assertTrue(launches.called)
                        self.assertEqual(launches.call_args.args[0].block, 64)
                    outputs = got if isinstance(got, tuple) else (got,)
                    results.append(tuple(t.clone().view(torch.uint8) for t in outputs))
                    x.add_(1)
                    graph.replay()
                    torch.cuda.synchronize(device)
                    with native._unconditional_masked(), dsl.disabled():
                        changed_reference = fn(x, **kwargs)
                    self.assertEqual(got, changed_reference, atol=1e-4, rtol=rtol)
                self.assertTrue(
                    row_select.called if stride == 1 else general_select.called
                )
                self.assertEqual(results[0], results[2])
                self.assertEqual(results[1], results[3])
                if op != "sum":
                    self.assertEqual(results[0], results[1])
        finally:
            native.set_aot_enabled(old_aot)
            if not old_enabled:
                dsl.disable()

    def test_required_inner_tree_cannot_fall_back(self, device):
        import cutlass

        from torch._native.ops.reductions import traits

        x = torch.empty((8, 2048), device=device)[:, ::2]
        with (
            mock.patch.object(kg, "_try_indexed_itree", return_value=None),
            mock.patch.object(kg, "_launch") as launch,
        ):
            with self.assertRaisesRegex(ValueError, "inner-tree.*cannot"):
                kg.reduce_dim(
                    traits.SumOps(acc=cutlass.Float32),
                    "sum",
                    x,
                    [1],
                    torch.float32,
                    order="inner_tree",
                )
            launch.assert_not_called()

    @parametrize("stride", [1, 2])
    def test_explicit_block_and_order_override_environment(self, device, stride):
        import cutlass

        from torch._native.ops.reductions import traits

        x = torch.randn((1024, 1024 * stride), device=device)[:, ::stride]
        with (
            torch._native._unconditional_masked(),
            torch.backends.python_native.cutedsl.disabled(),
        ):
            expected = x.sum(1)
        with (
            mock.patch.dict(os.environ, {rt._INNER_TREE_ENV: "1"}),
            mock.patch.object(rt, "select_row_order") as row_select,
            mock.patch.object(kg, "select_general_config") as general_select,
            mock.patch.object(kg, "_try_indexed_itree") as indexed,
            mock.patch.object(rt, "_run_itree") as row_tree,
        ):
            got = kg.reduce_dim(
                traits.SumOps(acc=cutlass.Float32),
                "sum",
                x,
                [1],
                torch.float32,
                block=64,
                order="unordered",
            )
            row_select.assert_not_called()
            general_select.assert_not_called()
            indexed.assert_not_called()
            row_tree.assert_not_called()
        self.assertEqual(got, expected, atol=1e-4, rtol=1e-4)


instantiate_parametrized_tests(TestReductionConfig)
instantiate_device_type_tests(TestReductionConfigDevice, globals(), only_for="cuda")


if __name__ == "__main__":
    run_tests()
