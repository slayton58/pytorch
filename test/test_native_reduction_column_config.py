# Owner(s): ["module: dsl-native-ops"]

import sys
import unittest
from unittest.mock import patch

import torch
from torch import _native as native
from torch.testing._internal.common_device_type import (
    dtypes,
    instantiate_device_type_tests,
    onlyCUDA,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TEST_CUTEDSL,
    TestCase,
)


if not TEST_CUTEDSL:
    if __name__ == "__main__":
        sys.exit(0)
    raise unittest.SkipTest("CuTeDSL not available")

import cutlass

from torch._native.ops.reductions import kernel_coltile as ct, traits as T


class TestColumnConfig(TestCase):
    def select(self, dtype=torch.float32, size=256, columns=4096, **kwargs):
        itemsize = 2 if dtype == torch.bfloat16 else 4
        args = dict(
            cc=(10, 7),
            dtype=dtype,
            rows=(size << 20) // (columns * itemsize),
            columns=columns,
            batches=1,
            nfields=1,
            trait_key="sum",
            itemsize=itemsize,
            acc_bits=32,
            nouts=1,
            alignment=16,
            contiguous=True,
        )
        args.update(kwargs)
        return ct.select_col_config(**args)

    @parametrize("dtype", [torch.float32, torch.bfloat16])
    @parametrize("size", [256, 2048])
    @parametrize("columns", [256, 1024])
    @parametrize(
        "trait_key,nfields",
        [
            ("sum", 1),
            ("mean", 1),
            ("amax", 1),
            ("argmaxi32", 2),
            ("var0", 3),
            ("vnorm2", 1),
        ],
    )
    def test_narrow_anchors(self, dtype, size, columns, trait_key, nfields):
        cfg = self.select(dtype, size, columns, trait_key=trait_key, nfields=nfields)
        self.assertEqual(cfg.threads_per_block, 64)
        self.assertEqual(cfg.vec, 4 if columns == 256 else 8)
        self.assertEqual(cfg.npar, 1024 if columns == 1024 and size == 256 else 4096)
        self.assertEqual(cfg.partial_layout, "column")
        self.assertEqual(cfg.combine_columns, 0)

    @parametrize(
        "dtype,size,trait_key,nfields,expected",
        [
            (torch.float32, 256, "sum", 1, (256, 4, "partition", 16)),
            (torch.float32, 2048, "sum", 1, (2048, 4, "partition", 8)),
            (torch.bfloat16, 256, "sum", 1, (512, 4, "partition", 8)),
            (torch.bfloat16, 2048, "sum", 1, (1024, 8, "column", 0)),
            (torch.float32, 256, "var0", 3, (256, 8, "column", 0)),
            (torch.float32, 2048, "var0", 3, (1024, 8, "column", 0)),
            (torch.bfloat16, 256, "var0", 3, (256, 8, "column", 0)),
            (torch.bfloat16, 2048, "var0", 3, (2048, 8, "column", 0)),
        ],
    )
    def test_wide_anchors(self, dtype, size, trait_key, nfields, expected):
        cfg = self.select(dtype, size, trait_key=trait_key, nfields=nfields)
        self.assertEqual(
            (cfg.npar, cfg.vec, cfg.partial_layout, cfg.combine_columns), expected
        )
        self.assertEqual(cfg.threads_per_block, 64)
        self.assertEqual(cfg.unroll, 4)

    @parametrize(
        "kwargs",
        [
            {"cc": (8, 0)},
            {"cc": (9, 0)},
            {"cc": (10, 0)},
            {"cc": (11, 0)},
            {"dtype": torch.float16},
            {"dtype": torch.float64},
            {"acc_bits": 64},
            {"nfields": 2},
            {"nouts": 2},
            {"batches": 2},
            {"alignment": 8},
            {"contiguous": False},
            {"trait_key": "var1"},
            {"trait_key": "varmean0"},
            {"trait_key": "prod"},
            {"trait_key": "argmaxi64"},
        ],
    )
    def test_unsupported_problem_retains_defaults(self, kwargs):
        cfg = self.select(**kwargs)
        self.assertEqual(cfg.rule, "default")
        self.assertEqual(cfg.combine_columns, 0)

    @parametrize("size", [16, 64, 255, 257, 1024, 2047, 2049])
    @parametrize("columns", [256, 1024, 4096])
    def test_unmeasured_sizes_retain_defaults(self, size, columns):
        self.assertEqual(self.select(size=size, columns=columns).rule, "default")

    @parametrize("columns", [255, 257, 512, 1023, 1025, 2048, 4095, 4097])
    def test_unmeasured_widths_retain_defaults(self, columns):
        self.assertEqual(self.select(columns=columns).rule, "default")

    @parametrize("kwargs", [{"threads_per_block": 128}, {"npar": 7}, {"vec": 2}])
    def test_explicit_geometry_disables_policy(self, kwargs):
        cfg = self.select(**kwargs)
        self.assertEqual(cfg.rule, "explicit")
        for key, value in kwargs.items():
            self.assertEqual(getattr(cfg, key), value)
        self.assertEqual(cfg.partial_layout, "column")
        self.assertEqual(cfg.combine_columns, 0)

    def test_inner_tree_never_selects_unordered_plan(self):
        self.assertIsNone(self.select(order="inner_tree"))
        with self.assertRaisesRegex(ValueError, "unknown reduction order"):
            self.select(order="linear")
        with self.assertRaisesRegex(ValueError, "requires unordered"):
            ct.reduce_col_tile(None, "sum", None, torch.float32, order="inner_tree")
        with self.assertRaisesRegex(ValueError, "require inner_tree"):
            ct.reduce_ordered_col(
                None, "sum", None, [torch.float32], 1, order="unordered"
            )

    def test_cache_distinguishes_order_and_hardware(self):
        ct.select_col_config.cache_clear()
        tuned = self.select()
        self.assertIsNone(self.select(order="inner_tree"))
        default = self.select(cc=(10, 0))
        self.assertEqual(default.rule, "default")
        self.assertIs(self.select(), tuned)
        self.assertIs(self.select(cc=(10, 0)), default)
        self.assertIsNone(self.select(order="inner_tree"))
        self.assertEqual(ct.select_col_config.cache_info().hits, 3)


class TestColumnCombine(TestCase):
    @onlyCUDA
    @dtypes(torch.float32, torch.bfloat16)
    @parametrize("op", ["sum", "argmax", "var", "var_mean"])
    @parametrize("pattern", ["signed", "offset", "constant", "nonfinite_ties"])
    @parametrize("tile_columns", [8, 16, 32])
    def test_tiled_combine_and_graph_replay(
        self, device, dtype, op, pattern, tile_columns
    ):
        x = torch.randn((257, 260), device=device, dtype=dtype)
        if pattern == "offset":
            x.mul_(8).add_(1024)
        elif pattern == "constant":
            x.fill_(3)
        elif pattern == "nonfinite_ties":
            x.fill_(0)
            x[0, :] = 1
            x[-1, :] = 1
            x[0, 0] = float("nan")
            x[1, 1] = float("inf")
            x[2, 2] = -float("inf")
        traits = {
            "sum": T.SumOps,
            "argmax": T.ArgMaxOps,
            "var": T.WelfordOps,
            "var_mean": T.VarMeanOps,
        }
        kw = {"correction": 0} if op in ("var", "var_mean") else {}
        trait = traits[op](acc=cutlass.Float32, **kw)
        nouts = 2 if op == "var_mean" else 1
        odt = torch.int64 if op == "argmax" else torch.float32 if op == "sum" else dtype
        cfg = ct.ColConfig(
            "unordered", "test_tiled", 64, 7, 4, "partition", tile_columns
        )

        def reference():
            with (
                native._unconditional_masked(),
                torch.backends.python_native.cutedsl.disabled(),
            ):
                kw = {"correction": 0} if op in ("var", "var_mean") else {}
                if op == "sum":
                    kw["dtype"] = torch.float32
                result = getattr(torch, op)(x, dim=0, **kw)
            return tuple(result) if nouts == 2 else (result,)

        def run():
            return ct._reduce_col_tile(
                trait,
                f"test_tiled_{op}",
                x,
                [odt] * nouts,
                nouts,
                None,
                None,
                None,
            )

        tol = 0 if op == "argmax" else 0.016 if dtype == torch.bfloat16 else 1e-4
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with patch.object(ct, "select_col_config", return_value=cfg):
            with torch.cuda.stream(stream):
                got = run()
            torch.cuda.current_stream().wait_stream(stream)
            self.assertEqual(got, reference(), rtol=tol, atol=tol, equal_nan=True)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                got = run()
        x.fill_(5)
        graph.replay()
        self.assertEqual(got, reference(), rtol=tol, atol=tol, equal_nan=True)


instantiate_parametrized_tests(TestColumnConfig)
instantiate_device_type_tests(TestColumnCombine, globals(), only_for=("cuda",))


if __name__ == "__main__":
    run_tests()
