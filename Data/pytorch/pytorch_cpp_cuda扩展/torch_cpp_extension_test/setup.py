import glob
import os.path as osp
from setuptools import setup, Extension
from torch.utils.cpp_extension import CppExtension,CUDAExtension,BuildExtension

ROOT_DIR = osp.dirname(osp.abspath(__file__))
include_dirs = [osp.join(ROOT_DIR,"include")]
source = glob.glob('*.cpp')+glob.glob('*.cu')


# cpp 扩展
# setup(
# name="cpp_interpolation",
# ext_modules=[CppExtension(
#     name="cpp_interpolation",
#     sources=["interpolation.cpp"],
# )],
# cmdclass={'build_ext':BuildExtension}
# )

# cuda 扩展
setup(
    name="cpp_cuda_interpolation",
    version = '1.0',
    ext_modules=[CUDAExtension(
        name="cpp_cuda_interpolation", #setup完成后 import使用的名称
        sources=source,
        include_dirs=include_dirs,
        # extra_compile_args= { #减少代码大小跟运行速度无关
        #     'cxx':['-02'],
        #     'nvcc':['-02']
        # }
    )],
    cmdclass={'build_ext':BuildExtension}
)
