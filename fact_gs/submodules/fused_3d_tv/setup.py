import os
import sys

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

os.environ.setdefault("PYTHONUNBUFFERED", "1")
sys.stderr.reconfigure(line_buffering=True)


def _parse_architectures():
    """Return normalized architecture strings (e.g. ['80', '86'])."""
    raw = None
    for env_key in ("CUDA_ARCHITECTURES", "TORCH_CUDA_ARCH_LIST"):
        raw = os.environ.get(env_key)
        if raw:
            break

    if not raw:
        return []

    arches = []
    for token in raw.replace(";", ",").split(","):
        token = token.strip()
        if not token:
            continue
        token = token.replace("+PTX", "")
        token = token.replace(".", "")
        arches.append(token)
    return arches


def get_extension():
    extra_compile_args = {
        "cxx": ["-O3", "-std=c++17"],
        "nvcc": ["-O3", "--use_fast_math"],
    }

    arches = _parse_architectures()
    if not arches:
        arches = ["75", "80", "90"]
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", ",".join(arches))

    for arch in arches:
        extra_compile_args["nvcc"].append(
            f"-gencode=arch=compute_{arch},code=sm_{arch}"
        )

    return CUDAExtension(
        name="fused_3d_tv_cuda",
        sources=["tv3d.cu", "ext.cpp"],
        extra_compile_args=extra_compile_args,
    )


setup(
    name="fused-3d-tv",
    packages=["fused_3d_tv"],
    ext_modules=[get_extension()],
    cmdclass={"build_ext": BuildExtension},
)
