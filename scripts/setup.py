import os
import pybind11
from setuptools import setup, Extension

ext_modules = [
    Extension(
        'rk4_backend',
        ['src/cpp/rk4_engine.cpp'],
        include_dirs=[
            pybind11.get_include(),
            os.path.join(os.path.dirname(pybind11.__file__), '..', 'numpy', 'core', 'include')
        ],
        language='c++',
        extra_compile_args=['-O3'],
    ),
]

setup(
    name='rk4_backend',
    version='1.0',
    description='C++ Accelerated RK4 Integration Engine',
    ext_modules=ext_modules,
)