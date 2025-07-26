from setuptools import setup
from torch.utils import cpp_extension

ext_modules = [
    cpp_extension.CUDAExtension(
        "swiftllm_c",
        [
            "src/entrypoints.cpp",
            "src/block_swapping.cpp",
        ],
        extra_compile_args={
            "cxx": ["-O3"],
            "nvcc": ["-O3", "--use_fast_math"],
        },
    ),
]

setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": cpp_extension.BuildExtension},
)
