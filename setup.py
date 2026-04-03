"""Build the ``gather_kv_cuda`` PyTorch CUDA extension (install-time, not JIT)."""

import os

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

_ROOT = os.path.dirname(os.path.abspath(__file__))

setup(
    name="kv-transfer-gather-cuda",
    version="0.1.0",
    ext_modules=[
        CUDAExtension(
            name="gather_kv_cuda",
            sources=[
                os.path.join(_ROOT, "csrc", "gather_kv.cpp"),
                os.path.join(_ROOT, "csrc", "gather_kv_cuda.cu"),
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "--use_fast_math"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
