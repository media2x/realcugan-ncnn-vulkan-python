#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
import pathlib

import cmake_build_extension
import setuptools

cmake_flags = [
    "-DBUILD_SHARED_LIBS:BOOL=OFF",
    "-DCALL_FROM_SETUP_PY:BOOL=ON",
]
cmake_flags.extend(os.environ.get("CMAKE_FLAGS", "").split())

setuptools.setup(
    ext_modules=[
        cmake_build_extension.CMakeExtension(
            name="realcugan-ncnn-vulkan-python",
            install_prefix="realcugan_ncnn_vulkan_python",
            write_top_level_init=(
                "from .realcugan_ncnn_vulkan import Realcugan, RealCUGAN, wrapped"
            ),
            source_dir=str(
                pathlib.Path(__file__).parent / "realcugan_ncnn_vulkan_python"
            ),
            cmake_configure_options=cmake_flags,
        )
    ],
    cmdclass={"build_ext": cmake_build_extension.BuildExtension},
)
