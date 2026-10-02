# Owner(s): ["module: dsl-native-ops"]

import sys
import unittest
from unittest.mock import patch

import torch
from torch import _native as native
from torch.testing._internal.common_device_type import (
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
            (torch.float32, 256, "sum", 1, (256, 8, "partition", 16)),
            (torch.float32, 2048, "sum", 1, (1024, 4, "partition", 8)),
            (torch.bfloat16, 256, "sum", 1, (256, 8, "partition", 16)),
            (torch.bfloat16, 2048, "sum", 1, (2048, 8, "partition", 8)),
            (torch.float32, 256, "argmaxi32", 2, (512, 4, "partition", 8)),
            (torch.float32, 2048, "argmaxi32", 2, (1024, 4, "partition", 8)),
            (torch.bfloat16, 256, "argmaxi32", 2, (512, 4, "partition", 8)),
            (torch.bfloat16, 2048, "argmaxi32", 2, (2048, 4, "partition", 8)),
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

    @parametrize("dtype", [torch.float32, torch.bfloat16])
    @parametrize("size", [256, 2048])
    @parametrize("key", ["mean", "amax", "vnorm2"])
    def test_single_field_column_geometry(self, dtype, size, key):
        expected = self.select(dtype, size)
        cfg = self.select(dtype, size, trait_key=key)
        self.assertEqual(cfg._replace(rule=expected.rule), expected)
        self.assertEqual(cfg.kernel_order, "linear")

    @parametrize("dtype", [torch.float32, torch.bfloat16])
    @parametrize("size", [256, 2048])
    @parametrize("columns", [1024, 4096])
    def test_welford_geometry_independent_of_output_count(self, dtype, size, columns):
        expected = self.select(dtype, size, columns, trait_key="var0", nfields=3)
        actual = self.select(
            dtype, size, columns, trait_key="varmean0", nfields=3, nouts=2
        )
        self.assertEqual(actual, expected)
        if columns == 4096:
            self.assertEqual(actual.kernel_order, "inner_tree")

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

    @parametrize(
        "dtype,size,columns,key,fields,nouts,expected",
        [
            (torch.float32, 16, 4096, "var0", 3, 1, "inner_tree"),
            (torch.float32, 64, 4096, "var0", 3, 1, "linear"),
            (torch.float32, 256, 256, "mean", 1, 1, "inner_tree"),
            (torch.bfloat16, 256, 256, "mean", 1, 1, "linear"),
            (torch.bfloat16, 64, 1024, "vnorm2", 1, 1, "inner_tree"),
            (torch.bfloat16, 256, 1024, "varmean0", 3, 2, "inner_tree"),
            (torch.bfloat16, 2048, 4096, "argmaxi32", 2, 1, "inner_tree"),
            (torch.bfloat16, 2048, 4096, "sum", 1, 1, "linear"),
        ],
    )
    def test_unordered_column_selection(
        self, dtype, size, columns, key, fields, nouts, expected
    ):
        cfg = self.select(
            dtype, size, columns, trait_key=key, nfields=fields, nouts=nouts
        )
        self.assertEqual(cfg.order, "unordered")
        self.assertEqual(cfg.kernel_order, expected)

    @parametrize(
        "updates",
        [
            {},
            {"cc": (8, 0)},
            {"cc": (9, 0)},
            {"cc": (10, 0)},
            {"cc": (11, 0)},
            {"rows": 1023},
            {"rows": 1025},
            {"columns": 4095},
            {"columns": 4097},
            {"threads_per_block": 64},
            {"npar": 16},
            {"vec": 4},
            {"alignment": 8},
            {"acc_bits": 64},
            {"contiguous": False},
            {"batches": 2},
            {"dtype": torch.float16},
            {"trait_key": "var1"},
            {"nfields": 2},
            {"nouts": 2},
        ],
    )
    def test_column_order_selection_guards(self, updates):
        args = dict(size=16, trait_key="var0", nfields=3)
        args.update(updates)
        self.assertEqual(
            self.select(**args).kernel_order, "linear" if updates else "inner_tree"
        )

    @parametrize(
        "key,output,pack",
        [
            ("amax", torch.bfloat16, 2),
            ("mean", torch.float32, 4),
            ("vnorm2", torch.float32, 4),
        ],
    )
    @parametrize(
        "updates",
        [
            {},
            {"cc": (8, 0)},
            {"cc": (9, 0)},
            {"cc": (10, 0)},
            {"cc": (11, 0)},
            {"rows": 262143},
            {"rows": 262145},
            {"columns": 4092},
            {"columns": 4100},
            {"batches": 2},
            {"dtype": torch.float32},
            {"field_bits": (64,)},
            {"out_dtypes": (torch.float64,)},
            {"full_tiles": False},
        ],
    )
    def test_ordered_column_packing_guards(self, key, output, pack, updates):
        args = dict(
            cc=(10, 7),
            dtype=torch.bfloat16,
            trait_key=key,
            rows=262144,
            columns=4096,
            batches=1,
            partials=32,
            field_bits=(32,),
            out_dtypes=(output,),
            full_tiles=True,
        )
        args.update(updates)
        cfg = ct.select_ordered_col_config(**args)
        if updates:
            self.assertTrue(cfg is None or cfg.column_pack == 1)
        else:
            self.assertEqual(
                cfg,
                ct.OrderedColConfig(
                    32, tile_columns=32 // pack, full_tiles=True, column_pack=pack
                ),
            )

    @parametrize(
        "key,fields,nouts,rows,columns,width",
        [
            ("sum", 1, 1, 65536, 129, 16),
            ("argmaxi32", 2, 1, 131072, 257, 32),
            ("var0", 3, 1, 16384, 256, 16),
            ("varmean0", 3, 2, 16384, 1024, 16),
            ("vnorm2", 1, 1, 65536, 1024, 16),
        ],
    )
    def test_complete_column_selection(self, key, fields, nouts, rows, columns, width):
        output = torch.int64 if fields == 2 else torch.float32
        cfg = ct.select_ordered_col_config(
            (10, 7),
            torch.float32,
            key,
            columns,
            1,
            max(1, rows // 8192),
            field_bits=(32,) * fields,
            out_dtypes=(output,) * nouts,
            rows=rows,
            full_tiles=True,
        )
        self.assertIsNotNone(cfg)
        self.assertEqual(
            (cfg.tile_columns, cfg.full_tiles, cfg.column_pack), (width, True, 1)
        )


class TestColumnCombine(TestCase):
    @onlyCUDA
    @parametrize(
        "dtype,op,pattern,tile_columns",
        [
            (torch.float32, "sum", "signed", 8),
            (torch.bfloat16, "mean", "offset", 16),
            (torch.bfloat16, "amax", "nonfinite_ties", 32),
            (torch.float32, "norm2", "signed", 16),
            (torch.float32, "argmax", "nonfinite_ties", 8),
            (torch.bfloat16, "var", "constant", 16),
            (torch.float32, "var_mean", "offset", 32),
        ],
    )
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
            "mean": T.MeanOps,
            "amax": T.AMaxOps,
            "norm2": lambda **kw: T.NormOps(2, **kw),
            "argmax": T.ArgMaxOps,
            "var": T.WelfordOps,
            "var_mean": T.VarMeanOps,
        }
        kw = {"correction": 0} if op in ("var", "var_mean") else {}
        trait = traits[op](acc=cutlass.Float32, **kw)
        nouts = 2 if op == "var_mean" else 1
        odt = (
            torch.int64
            if op == "argmax"
            else torch.float32
            if op in ("sum", "mean", "norm2")
            else dtype
        )
        cfg = ct.ColConfig(
            "unordered", "test_tiled", 64, 7, 4, "partition", tile_columns
        )

        def reference():
            with (
                native._unconditional_masked(),
                torch.backends.python_native.cutedsl.disabled(),
            ):
                kw = {"correction": 0} if op in ("var", "var_mean") else {}
                if op in ("sum", "mean", "norm2"):
                    kw["dtype"] = torch.float32
                result = (
                    torch.linalg.vector_norm(x, ord=2, dim=0, **kw)
                    if op == "norm2"
                    else getattr(torch, op)(x, dim=0, **kw)
                )
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
