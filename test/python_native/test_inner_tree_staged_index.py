# Owner(s): ["module: dsl-native-ops"]

import unittest
from unittest import mock

import torch
from torch.testing._internal.common_cuda import SM90OrLater, TEST_CUDA
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import (
    parametrize,
    run_tests,
    TEST_CUTEDSL,
    TestCase,
)


@unittest.skipUnless(
    TEST_CUDA and SM90OrLater and TEST_CUTEDSL, "requires CUDA and CuTeDSL"
)
class TestInnerTreeStagedIndex(TestCase):
    @parametrize("dtype", [torch.float32, torch.bfloat16])
    @parametrize("op", ["argmax", "argmin"])
    @parametrize("pattern", ["unique", "ties", "nan", "constant"])
    @parametrize("rows", [1, 3])
    def test_split_batch_indices(self, device, dtype, op, pattern, rows):
        import cutlass

        from torch._native.ops.reductions import kernel_rowtile as rt, traits

        n = 262144
        x = torch.zeros((rows, n), dtype=dtype, device=device)
        trait_type = traits.ArgMaxOps if op == "argmax" else traits.ArgMinOps
        trait = trait_type(acc=cutlass.Float32, idx=cutlass.Int32)
        key = f"{op}i32"
        cc = torch.cuda.get_device_capability(device)
        policy = rt._ITREE_ARCH["default"]._replace(split_stage_rows=4)
        with mock.patch.dict(rt._ITREE_ARCH, {cc: policy}):
            staged = rt.trait_itree_plan(
                trait, n, rows, x.element_size(), device=device
            )
            self.assertEqual(staged.shape, "split")
            self.assertEqual(staged.rows_per_block, 4)
            self.assertGreater(staged.stage_e, 0)
            unstaged = staged._replace(rows_per_block=1, stage_e=0)
            batch_size = staged.split[1]
            value = 1 if op == "argmax" else -1
            for row in range(rows):
                first = (row + 1) * batch_size + 123
                if pattern != "constant":
                    x[row, first] = float("nan") if pattern == "nan" else value
                if pattern in ("ties", "nan"):
                    # A later batch has a smaller local index but must lose the tie.
                    x[row, first + batch_size - 116] = x[row, first]

            with torch._native._unconditional_masked():
                expected = getattr(torch, op)(x, dim=1)
            stream = torch.cuda.Stream(device=device)
            stream.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.stream(stream):
                direct = rt._run_itree(trait, key, x, [torch.int64], unstaged)[0]
                rt._run_itree(trait, key, x, [torch.int64], staged)
            stream.synchronize()
            self.assertEqual(direct, expected)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                actual = rt._run_itree(trait, key, x, [torch.int64], staged)[0]
            graph.replay()
            torch.cuda.synchronize(device)
            self.assertEqual(actual, expected)

            x.copy_(x.flip(1))
            with torch._native._unconditional_masked():
                expected = getattr(torch, op)(x, dim=1)
            graph.replay()
            torch.cuda.synchronize(device)
            self.assertEqual(actual, expected)
            self.assertEqual(
                rt._run_itree(trait, key, x, [torch.int64], unstaged)[0], expected
            )


instantiate_device_type_tests(TestInnerTreeStagedIndex, globals(), only_for="cuda")


if __name__ == "__main__":
    run_tests()
