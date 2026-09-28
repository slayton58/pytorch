# Owner(s): ["module: dsl-native-ops"]

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
class TestInnerTreeCombineUnroll(TestCase):
    @parametrize("unroll", [0, 1, 4, 8, 16, 32, 64, 128, 4096])
    @parametrize("rows", [1, 33])
    def test_unroll_preserves_staging_geometry_and_keys(self, unroll, rows):
        from torch._native.ops.reductions import kernel_rowtile as rt

        policy = rt._ITREE_ARCH[(10, 0)]
        split = rt._ItreePlan(
            "split", 4, 16, 4, 2, (), (), (8192, 8192, 8192, 512, 512)
        )
        caps = SimpleNamespace(smem_per_block_optin=232448)
        with mock.patch.object(rt._hw, "caps", return_value=caps):
            with mock.patch.object(rt, "_itree_arch", return_value=policy):
                baseline = rt.itree_combine_plan(split, 4, nfields=3, nrows=rows)
            with mock.patch.object(
                rt, "_itree_arch", return_value=policy._replace(combine_unroll=unroll)
            ):
                plan = rt.itree_combine_plan(split, 4, nfields=3, nrows=rows)
        self.assertGreater(plan.combine_tile, 0)
        self.assertEqual(plan.combine_unroll, min(unroll, plan.combine_tile))
        self.assertEqual(plan._replace(combine_unroll=0), baseline)
        self.assertEqual(plan.sig == baseline.sig, unroll == 0)

    def test_unused_unroll_does_not_change_unstaged_key(self):
        from torch._native.ops.reductions import kernel_rowtile as rt

        split = rt._ItreePlan(
            "split", 4, 16, 1, 2, (), (), (8192, 8192, 8192, 512, 512)
        )
        policy = rt._ITREE_ARCH["default"]._replace(combine_unroll=8)
        with (
            mock.patch.object(rt, "_itree_arch", return_value=policy),
            mock.patch.object(
                rt._hw,
                "caps",
                return_value=SimpleNamespace(smem_per_block_optin=232448),
            ),
        ):
            plan = rt.itree_combine_plan(split, 4, nrows=1)
        self.assertEqual((plan.combine_tile, plan.combine_unroll), (0, 0))

    def test_invalid_unroll(self):
        from torch._native.ops.reductions import kernel_rowtile as rt

        with mock.patch.object(
            rt,
            "_itree_arch",
            return_value=rt._ITREE_ARCH[(10, 0)]._replace(combine_unroll=-1),
        ):
            with self.assertRaisesRegex(ValueError, "combine_unroll"):
                rt.itree_combine_plan(None, 4)

    def test_fake_descriptors_preserve_per_field_alignment(self):
        from torch._native.ops.reductions import kernel_general as kg

        tensors = [
            SimpleNamespace(dtype=torch.float32),
            SimpleNamespace(dtype=torch.int64),
        ]
        with mock.patch.object(kg._L, "fake_compact") as fake:
            kg._fakes(tensors, int64_extent=True, alignments=(16, 8))
        self.assertEqual(
            [call.kwargs["align"] for call in fake.call_args_list], [16, 8]
        )


@unittest.skipUnless(
    TEST_CUDA and SM90OrLater and TEST_CUTEDSL, "requires CUDA and CuTeDSL"
)
class TestInnerTreeCombineUnrollDevice(TestCase):
    def test_unaligned_partials_rejected_with_cached_plan(self, device):
        import cutlass

        from torch._native.ops.reductions import (
            kernel_general as kg,
            kernel_rowtile as rt,
            traits,
        )

        plan = rt._ItreePlan(
            "combine",
            1,
            0,
            1,
            2,
            (),
            (),
            (512, 8192, 8192, 512, 512),
            combine_grp=4,
            combine_tile=512,
            combine_unroll=4,
        )
        op = kg.ReduceBlock(
            traits.SumOps(acc=cutlass.Float32),
            count=512,
            num_o=1,
            red_pairs=[],
            kept_pairs=[],
            order="inner_tree",
            itree=plan,
        )
        aligned = torch.ones(512, device=device)
        unaligned = torch.ones(513, device=device)[1:]
        output = torch.empty(1, device=device)
        key = ("test_unroll_alignment", "sum") + op.cache_sig
        kg._launch(op, key, [aligned], [output])
        torch.cuda.synchronize(device)
        self.assertEqual(output, torch.full_like(output, 512))
        with self.assertRaisesRegex(ValueError, "aligned partial buffers"):
            kg._launch(op, key, [unaligned], [output])

    @parametrize("op", ["sum", "var", "var_mean", "argmax"])
    @parametrize("rows", [1, 33])
    @parametrize("partials", [2048, 4096])
    @parametrize("pattern", ["signed", "special"])
    def test_staged_fold_bits_and_graph_replay(
        self, device, op, rows, partials, pattern
    ):
        import cutlass

        from torch._native.ops.reductions import (
            kernel_general as kg,
            kernel_rowtile as rt,
            traits,
        )

        values = torch.randn((rows, partials), device=device)
        if op == "sum":
            trait, out_dtypes = traits.SumOps(acc=cutlass.Float32), [torch.float32]
            if pattern == "special":
                values.fill_(-0.0)
            parts = [values.flatten()]
        elif op == "argmax":
            trait, out_dtypes = traits.ArgMaxOps(acc=cutlass.Float32), [torch.int64]
            if pattern == "special":
                values.zero_()
                values[:, 123] = float("nan")
                values[:, -7] = float("nan")
            indices = torch.arange(partials, device=device, dtype=torch.int32).repeat(
                rows
            )
            parts = [values.flatten(), indices]
        else:
            trait_type = traits.VarMeanOps if op == "var_mean" else traits.WelfordOps
            trait = trait_type(correction=0, acc=cutlass.Float32)
            out_dtypes = [torch.float32] * (2 if op == "var_mean" else 1)
            counts = torch.ones_like(values)
            if pattern == "special":
                values.mul_(8).add_(1024)
                counts[:, ::7] = 0
            parts = [
                values.flatten(),
                torch.zeros_like(values).flatten(),
                counts.flatten(),
            ]
        split = rt._ItreePlan(
            "split", 4, 16, 4, 2, (), (), (partials, 8192, 8192, 512, 512)
        )
        with mock.patch.object(rt, "_itree_arch", return_value=rt._ITREE_ARCH[(10, 0)]):
            staged = rt.itree_combine_plan(
                split, 4, device, nfields=trait.nfields, nrows=rows
            )
        self.assertGreater(staged.combine_tile, 0)
        baseline = staged._replace(combine_tile=0, combine_unroll=0)

        def launch(plan, outs):
            block = kg.ReduceBlock(
                trait,
                count=partials,
                num_o=rows,
                red_pairs=[],
                kept_pairs=[],
                project_n=partials,
                nouts=len(outs),
                order="inner_tree",
                itree=plan,
            )
            key = (
                "test_unroll_combine",
                op,
                tuple(out_dtypes),
                tuple(p.dtype for p in parts),
            ) + block.cache_sig
            kg._launch(block, key, parts, outs)

        reference = [
            torch.empty(rows, device=device, dtype=dtype) for dtype in out_dtypes
        ]
        for unroll in (1, 4, 8, 16, 32, 64, 128):
            with self.subTest(unroll=unroll):
                actual = [torch.empty_like(t) for t in reference]
                plan = staged._replace(combine_unroll=unroll)
                stream = torch.cuda.Stream(device=device)
                stream.wait_stream(torch.cuda.current_stream(device))
                with torch.cuda.stream(stream):
                    launch(baseline, reference)
                    launch(plan, actual)
                stream.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    launch(plan, actual)
                graph.replay()
                torch.cuda.synchronize(device)
                for got, expected in zip(actual, reference):
                    self.assertEqual(got.view(torch.uint8), expected.view(torch.uint8))
                original = parts[0].clone()
                parts[0].add_(0.125)
                launch(baseline, reference)
                graph.replay()
                torch.cuda.synchronize(device)
                for got, expected in zip(actual, reference):
                    self.assertEqual(got.view(torch.uint8), expected.view(torch.uint8))
                parts[0].copy_(original)


instantiate_parametrized_tests(TestInnerTreeCombineUnroll)
instantiate_device_type_tests(
    TestInnerTreeCombineUnrollDevice, globals(), only_for="cuda"
)


if __name__ == "__main__":
    run_tests()
