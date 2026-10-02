# Owner(s): ["module: dsl-native-ops"]

import os
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


@unittest.skipUnless(TEST_CUTEDSL, "requires CuTeDSL")
class TestFullInnerTreeConfig(TestCase):
    @parametrize("dtype", [torch.float32, torch.bfloat16])
    @parametrize("mib", [256, 2048])
    @parametrize(
        "key,fields,nouts,unroll",
        [
            ("sum", 1, 1, 0),
            ("argmaxi32", 2, 1, 128),
            ("var0", 3, 1, 64),
            ("varmean0", 3, 2, 64),
        ],
    )
    def test_measured_profiles_and_dag(self, dtype, mib, key, fields, nouts, unroll):
        from torch._native.ops.reductions import kernel_rowtile as rt

        itemsize = 4 if dtype == torch.float32 else 2
        N = (mib << 20) // itemsize
        output = (
            torch.float32
            if key == "sum"
            else torch.int64
            if key == "argmaxi32"
            else dtype
        )
        arch = rt.select_full_itree_arch(
            (10, 7),
            dtype,
            key,
            N,
            1,
            field_bits=(32,) * fields,
            out_dtypes=(output,) * nouts,
            alignment=16,
            contiguous=True,
        )
        self.assertEqual(
            arch,
            rt._ItreeArch(
                4,
                32,
                16,
                False,
                32,
                2048,
                unroll,
                uniform_count=fields == 3,
                combine_weights=fields == 3,
            ),
        )
        baseline = rt.itree_plan(N, 1, itemsize, arch=rt._ITREE_ARCH[(10, 7)])
        selected = rt.itree_plan(N, 1, itemsize, arch=arch)
        self.assertEqual(
            selected._replace(rows_per_block=1, stage_e=0).sig, baseline.sig
        )
        self.assertEqual((selected.rows_per_block, selected.stage_e), (4, 32))
        caps = SimpleNamespace(smem_per_block_optin=232448)
        with (
            mock.patch.object(rt._hw, "caps", return_value=caps),
            mock.patch.object(
                rt, "_itree_arch", side_effect=AssertionError("explicit arch lost")
            ),
        ):
            combine = rt.itree_combine_plan(
                selected, 4, nfields=fields, nrows=1, arch=arch
            )
        self.assertEqual(
            (combine.rows_per_block, combine.combine_grp, combine.combine_tile),
            (1, 4, 2048),
        )
        self.assertEqual(combine.combine_unroll, unroll)
        self.assertEqual(combine.combine_count, selected.split[1] if fields == 3 else 0)
        self.assertEqual(combine.combine_weights, fields == 3)
        self.assertEqual(combine.split, baseline.split)

    @parametrize(
        "updates",
        [
            {"cc": (8, 0)},
            {"cc": (9, 0)},
            {"cc": (10, 0)},
            {"cc": (10, 3)},
            {"cc": (11, 0)},
            {"dtype": torch.float16},
            {"dtype": torch.float64},
            {"N": (1 << 26) - 1},
            {"N": (1 << 26) + 1},
            {"N": (1 << 29) - 1},
            {"N": (1 << 29) + 1},
            {"N": 1 << 22},
            {"N": 1 << 27},
            {"N": 1 << 28},
            {"M": 2},
            {"M": 256},
            {"M": 1024},
            {"M": 4096},
            {"alignment": 8},
            {"alignment": 4},
            {"contiguous": False},
            {"field_bits": (64,)},
            {"field_bits": (32, 32)},
            {"out_dtypes": (torch.bfloat16,)},
            {"out_dtypes": (torch.float64,)},
            {"out_dtypes": (torch.float32, torch.float32)},
            {"trait_key": "mean"},
            {"trait_key": "nansum"},
            {"trait_key": "argmini32"},
            {"trait_key": "var1", "field_bits": (32, 32, 32)},
            {
                "trait_key": "varmean1",
                "field_bits": (32, 32, 32),
                "out_dtypes": (torch.float32, torch.float32),
            },
            {
                "trait_key": "argmaxi64",
                "field_bits": (32, 64),
                "out_dtypes": (torch.int64,),
            },
        ],
    )
    def test_unmeasured_inputs_keep_arch_default(self, updates):
        from torch._native.ops.reductions import kernel_rowtile as rt

        args = dict(
            cc=(10, 7),
            dtype=torch.float32,
            trait_key="sum",
            N=1 << 26,
            M=1,
            field_bits=(32,),
            out_dtypes=(torch.float32,),
            alignment=16,
            contiguous=True,
        )
        args.update(updates)
        self.assertIsNone(rt.select_full_itree_arch(**args))

    @parametrize("entry", ["tile", "out"])
    def test_selected_arch_reaches_both_stages(self, entry):
        from torch._native.ops.reductions import kernel_rowtile as rt

        x = SimpleNamespace(
            shape=(1, 1 << 26),
            dtype=torch.float32,
            device="cuda",
            is_cuda=True,
            dim=lambda: 2,
            stride=lambda axis: 1,
            is_contiguous=lambda: True,
            element_size=lambda: 4,
        )
        trait = SimpleNamespace(fdtypes=(SimpleNamespace(width=32),))
        output = SimpleNamespace(dtype=torch.float32)
        with (
            mock.patch.object(rt._hw, "caps", return_value=SimpleNamespace(cc=(10, 7))),
            mock.patch.object(rt._L, "supported_alignment", return_value=16),
            mock.patch.object(
                rt, "trait_itree_plan", wraps=rt.trait_itree_plan
            ) as plan,
            mock.patch.object(rt, "_run_itree", return_value=(output,)) as run,
        ):
            if entry == "tile":
                self.assertEqual(
                    rt.reduce_row_tile(
                        trait, "sum", x, [torch.float32], order="inner_tree"
                    ),
                    (output,),
                )
            else:
                self.assertTrue(rt.reduce_row_itree(trait, "sum", x, output))
        self.assertIsNotNone(plan.call_args.kwargs["arch"])
        self.assertEqual(plan.call_args.kwargs["arch"], run.call_args.kwargs["arch"])
        self.assertEqual(run.call_args.args[4].rows_per_block, 4)

    def test_linear_order_does_not_select_full_profile(self):
        from torch._native.ops.reductions import kernel_rowtile as rt

        x = SimpleNamespace(
            shape=(1, 1 << 26),
            is_cuda=True,
            dim=lambda: 2,
            stride=lambda axis: 1,
            element_size=lambda: 4,
            device="cuda",
        )
        with (
            mock.patch.object(rt, "_full_itree_arch") as select,
            mock.patch.object(
                rt, "trait_row_config", side_effect=RuntimeError("linear path")
            ),
            mock.patch.dict(os.environ, {rt._INNER_TREE_ENV: "1"}),
        ):
            with self.assertRaisesRegex(RuntimeError, "linear path"):
                rt.reduce_row_tile(None, "sum", x, [torch.float32], order="linear")
            select.assert_not_called()

    @parametrize(
        "count,last,itemsize,fields",
        [
            (8192, 8192, 4, 3),
            (8192, 8191, 4, 3),
            (8191, 8191, 4, 3),
            (8192, 8192, 8, 3),
            (8192, 8192, 4, 1),
        ],
    )
    @parametrize("weights", [False, True])
    def test_uniform_count_guard_and_cache_key(
        self, count, last, itemsize, fields, weights
    ):
        from torch._native.ops.reductions import kernel_rowtile as rt

        split = rt._ItreePlan(
            "split", 4, 16, 1, 2, (), (), (1024, count, last, 512, 512)
        )
        arch = rt._ITREE_ARCH["default"]._replace(
            combine_group_bytes=16,
            combine_async=8,
            combine_max=512,
            combine_unroll=4,
        )
        with mock.patch.object(
            rt._hw, "caps", return_value=SimpleNamespace(smem_per_block_optin=232448)
        ):
            baseline = rt.itree_combine_plan(
                split, itemsize, nfields=fields, nrows=1, arch=arch
            )
            candidate = rt.itree_combine_plan(
                split,
                itemsize,
                nfields=fields,
                nrows=1,
                arch=arch,
                uniform_count=True,
                combine_weights=weights,
            )
        supported = (count, last, itemsize, fields) == (8192, 8192, 4, 3)
        self.assertEqual(candidate.combine_count, count if supported else 0)
        self.assertEqual(candidate.combine_weights, weights and supported)
        self.assertEqual(
            candidate._replace(combine_count=0, combine_weights=False), baseline
        )
        self.assertEqual(candidate.sig != baseline.sig, supported)


@unittest.skipUnless(TEST_CUDA and TEST_CUTEDSL, "requires CUDA and CuTeDSL")
class TestFullInnerTreeConfigDevice(TestCase):
    @parametrize("dtype", [torch.float32, torch.bfloat16])
    @parametrize("op", ["sum", "argmax", "var", "var_mean"])
    @parametrize("dim", [None, 0])
    def test_public_dispatch_and_graph_replay(self, device, dtype, op, dim):
        from torch import _native as native
        from torch._native.ops.reductions import kernel_rowtile as rt

        if torch.cuda.get_device_capability(device) != (10, 7):
            self.skipTest("full-reduction profile is specific to Rubin")
        itemsize = 4 if dtype == torch.float32 else 2
        x = torch.empty((256 << 20) // itemsize, device=device, dtype=dtype)
        x.uniform_(0.25, 0.75)
        fn = getattr(torch, op)
        kwargs = {"dim": dim}
        if op == "sum":
            kwargs["dtype"] = torch.float32
        elif op in ("var", "var_mean"):
            kwargs["correction"] = 0
        dsl = torch.backends.python_native.cutedsl
        old_aot, old_enabled = native.aot_enabled(), dsl.enabled
        native.set_aot_enabled(False)
        dsl.enable()
        gates = {rt._SUM_INNER_TREE_ENV: "0", rt._INNER_TREE_ENV: "1"}
        try:
            with torch.inference_mode(), mock.patch.dict(os.environ, gates):
                with mock.patch.object(rt, "select_full_itree_arch", return_value=None):
                    baseline = fn(x, **kwargs)
                with native._unconditional_masked(), dsl.disabled():
                    aten = fn(x, **kwargs)
                stream = torch.cuda.Stream(device=device)
                stream.wait_stream(torch.cuda.current_stream(device))
                with torch.cuda.stream(stream):
                    fn(x, **kwargs)
                stream.synchronize()
                graph = torch.cuda.CUDAGraph()
                with mock.patch.object(
                    rt, "_launch_itree", wraps=rt._launch_itree
                ) as launches:
                    with torch.cuda.graph(graph, stream=stream):
                        got = fn(x, **kwargs)
                self.assertEqual(
                    [c.args[7] for c in launches.call_args_list],
                    ["rowitree1", "rowitree2"],
                )
                producer, combine = (c.args[2] for c in launches.call_args_list)
                self.assertEqual((producer.rows_per_block, producer.stage_e), (4, 32))
                self.assertEqual((combine.combine_grp, combine.combine_tile), (4, 2048))
                self.assertEqual(
                    combine.combine_count,
                    producer.split[1] if op in ("var", "var_mean") else 0,
                )
                self.assertEqual(combine.combine_weights, op in ("var", "var_mean"))
                self.assertEqual(
                    combine.combine_unroll,
                    0 if op == "sum" else 128 if op == "argmax" else 64,
                )
                for changed in (False, True):
                    if changed:
                        x.uniform_(0.5, 1.0)
                        with mock.patch.object(
                            rt, "select_full_itree_arch", return_value=None
                        ):
                            baseline = fn(x, **kwargs)
                        with native._unconditional_masked(), dsl.disabled():
                            aten = fn(x, **kwargs)
                    # Populating an unordered cache entry must not change a captured ordered launch.
                    with mock.patch.dict(os.environ, {rt._INNER_TREE_ENV: "0"}):
                        fn(x, **kwargs)
                    graph.replay()
                    torch.cuda.synchronize(device)
                    actuals = got if isinstance(got, tuple) else (got,)
                    references = (
                        baseline if isinstance(baseline, tuple) else (baseline,)
                    )
                    for actual, reference in zip(actuals, references):
                        self.assertEqual(
                            actual.reshape(-1).view(torch.uint8),
                            reference.reshape(-1).view(torch.uint8),
                        )
                    self.assertEqual(
                        got,
                        aten,
                        rtol=0.02 if dtype == torch.bfloat16 else 1e-4,
                        atol=1e-4,
                    )
        finally:
            native.set_aot_enabled(old_aot)
            if not old_enabled:
                dsl.disable()

    @unittest.skipUnless(SM90OrLater, "requires Hopper or later")
    @parametrize("weights", [False, True])
    def test_uniform_count_bits(self, device, weights):
        import cutlass

        from torch._native.ops.reductions import (
            kernel_general as kg,
            kernel_rowtile as rt,
            traits,
        )

        rows, partials, count = 33, 1024, 8192
        trait = traits.VarMeanOps(correction=1, acc=cutlass.Float32)
        means = torch.randn(rows, partials, device=device).mul_(8).add_(1024)
        means[0].fill_(-0.0)
        means[1, 17] = float("nan")
        means[2, 33] = float("inf")
        parts = [
            means.flatten(),
            torch.rand_like(means).flatten(),
            torch.full_like(means, count).flatten(),
        ]
        plan = rt._ItreePlan(
            "combine",
            1,
            0,
            1,
            2,
            (),
            (),
            (partials, count, count, 512, 512),
            combine_grp=4,
            combine_tile=512,
            combine_unroll=4,
        )
        baseline = [torch.empty(rows, device=device) for _ in range(2)]
        actual = [torch.empty_like(t) for t in baseline]

        def launch(selected, outs):
            block = kg.ReduceBlock(
                trait,
                count=partials,
                num_o=rows,
                red_pairs=[],
                kept_pairs=[],
                project_n=count * partials,
                nouts=2,
                order="inner_tree",
                itree=selected,
            )
            key = ("test_uniform_count",) + block.cache_sig
            kg._launch(block, key, parts, outs)

        candidate = plan._replace(combine_count=count, combine_weights=weights)
        launch(plan, baseline)
        launch(candidate, actual)
        for got, expected in zip(actual, baseline):
            self.assertEqual(got.view(torch.uint8), expected.view(torch.uint8))


instantiate_parametrized_tests(TestFullInnerTreeConfig)
instantiate_device_type_tests(TestFullInnerTreeConfigDevice, globals(), only_for="cuda")


if __name__ == "__main__":
    run_tests()
