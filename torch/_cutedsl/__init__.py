from . import scaled_grouped_mm_mxfp8 as mm_mxfp8

scaled_grouped_mm_mxfp8 = mm_mxfp8.scaled_grouped_mm_mxfp8
scaled_grouped_mm_mxfp8_register_kernels = mm_mxfp8._scaled_grouped_mm_mxfp8_register_kernels

scaled_grouped_mm_mxfp8.__module__ = __name__
scaled_grouped_mm_mxfp8_register_kernels.__module__ = __name__

__all__ = ["scaled_grouped_mm_mxfp8", "scaled_grouped_mm_mxfp8_register_kernels"]
