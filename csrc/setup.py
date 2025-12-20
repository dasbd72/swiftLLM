import os
import subprocess

from setuptools import setup
from torch.utils import cpp_extension

# Get the absolute path to the directory containing this setup.py file
project_root = os.path.dirname(os.path.abspath(__file__))


class BuildExtensionWithISPC(cpp_extension.BuildExtension):
    """A custom build extension class that integrates ISPC compilation."""

    def run(self):
        # Execute the command. check_call will raise an error if the command fails.
        try:
            subprocess.check_call(["bash", "build_ispc.sh"], cwd=project_root)
            print("ISPC compilation successful.")
        except subprocess.CalledProcessError as e:
            print(f"Compilation failed with error: {e}")
            exit(1)

        # Call the original run method to proceed with C++/CUDA compilation
        super().run()


ext_modules = [
    cpp_extension.CUDAExtension(
        "swiftllm_c",
        [
            "src/entrypoints.cpp",
            "src/block_swapping.cpp",
            "src/cpu_paged_attention.cpp",
            "src/cpu_kvcache_mgmt.cpp",
            "src/thread_pool.cpp",
        ],
        extra_compile_args={
            "cxx": [
                "-O3",
                "-fopenmp",
                "-DISPC_CPU_PAGED_ATTENTION",
                "-D__fp16=_Float16",
            ],
            "nvcc": ["-O3", "--use_fast_math"],
        },
        extra_objects=[
            "src/cpu_paged_attention_ispc.o",
        ],
    ),
]

setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtensionWithISPC},
)
