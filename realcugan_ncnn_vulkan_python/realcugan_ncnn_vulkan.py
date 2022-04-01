#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Name: RealCUGAN ncnn Vulkan Python wrapper
Author: K4YT3X
Date Created: March 19, 2022
Last Modified: March 19, 2022
"""

import importlib
import pathlib
import sys

from PIL import Image

if __package__ is None:
    import realcugan_ncnn_vulkan_wrapper as wrapped
else:
    wrapped = importlib.import_module(f"{__package__}.realcugan_ncnn_vulkan_wrapper")


class Realcugan:
    """
    Python FFI for RealCUGAN implemented with ncnn library

    :param gpuid int: gpu device to use (-1=cpu)
    :param tta_mode bool: enable test time argumentation
    :param num_threads int: processing thread count
    :param noise int: denoise level
    :param scale int: upscale ratio
    :param tilesize int: tile size
    :param syncgap int: sync gap mode
    :param model str: realcugan model name
    """

    def __init__(
        self,
        gpuid: int = 0,
        tta_mode: bool = False,
        num_threads: int = 1,
        noise: int = -1,
        scale: int = 2,
        tilesize: int = 0,
        syncgap: int = 3,
        model: str = "models-se",
        **_kwargs,
    ):
        # check arguments' validity
        assert gpuid >= -1, "gpuid must >= -1"
        assert noise in range(-1, 4), "noise must be -1-3"
        assert scale in range(1, 5), "scale must be 1-4"
        assert tilesize == 0 or tilesize >= 32, "tilesize must >= 32 or be 0"
        assert syncgap in range(4), "syncgap must be 0-3"
        assert num_threads >= 1, "num_threads must be a positive integer"

        self._realcugan_object = wrapped.RealCUGANWrapped(gpuid, tta_mode, num_threads)
        self._model = model
        self._gpuid = gpuid
        self._realcugan_object.noise = noise
        self._realcugan_object.scale = scale
        self._realcugan_object.tilesize = (
            self._get_tilesize() if tilesize <= 0 else tilesize
        )
        self._realcugan_object.prepadding = self._get_prepadding()
        self._realcugan_object.syncgap = syncgap
        self._load()

    def _load(
        self, param_path: pathlib.Path = None, model_path: pathlib.Path = None
    ) -> None:
        """
        Load models from given paths. Use self.model if one or all of the parameters are not given.

        :param parampath: the path to model params. usually ended with ".param"
        :param modelpath: the path to model bin. usually ended with ".bin"
        :return: None
        """
        if param_path is None or model_path is None:
            model_path = pathlib.Path(self._model)
            if not model_path.is_dir():
                model_path = pathlib.Path(__file__).parent / "models" / self._model

                if self._realcugan_object.noise == -1:
                    param_path = (
                        model_path
                        / f"up{self._realcugan_object.scale}x-conservative.param"
                    )
                    model_path = (
                        model_path
                        / f"up{self._realcugan_object.scale}x-conservative.bin"
                    )
                elif self._realcugan_object.noise == 0:
                    param_path = (
                        model_path
                        / f"up{self._realcugan_object.scale}x-no-denoise.param"
                    )
                    model_path = (
                        model_path / f"up{self._realcugan_object.scale}x-no-denoise.bin"
                    )
                else:
                    param_path = (
                        model_path
                        / f"up{self._realcugan_object.scale}x-denoise{self._realcugan_object.noise}x.param"
                    )
                    model_path = (
                        model_path
                        / f"up{self._realcugan_object.scale}x-denoise{self._realcugan_object.noise}x.bin"
                    )

        if param_path.exists() and model_path.exists():
            param_path_str, model_path_str = wrapped.StringType(), wrapped.StringType()
            if sys.platform in ("win32", "cygwin"):
                param_path_str.wstr = wrapped.new_wstr_p()
                wrapped.wstr_p_assign(param_path_str.wstr, str(param_path))
                model_path_str.wstr = wrapped.new_wstr_p()
                wrapped.wstr_p_assign(model_path_str.wstr, str(model_path))
            else:
                param_path_str.str = wrapped.new_str_p()
                wrapped.str_p_assign(param_path_str.str, str(param_path))
                model_path_str.str = wrapped.new_str_p()
                wrapped.str_p_assign(model_path_str.str, str(model_path))

            self._realcugan_object.load(param_path_str, model_path_str)
        else:
            raise FileNotFoundError(f"{param_path} or {model_path} not found")

    def process(self, image: Image) -> Image:
        """
        Process the incoming PIL.Image

        :param im: PIL.Image
        :return: PIL.Image
        """
        in_bytes = bytearray(image.tobytes())
        channels = int(len(in_bytes) / (image.width * image.height))
        out_bytes = bytearray((self._realcugan_object.scale ** 2) * len(in_bytes))

        raw_in_image = wrapped.Image(in_bytes, image.width, image.height, channels)
        raw_out_image = wrapped.Image(
            out_bytes,
            self._realcugan_object.scale * image.width,
            self._realcugan_object.scale * image.height,
            channels,
        )

        if self._gpuid != -1:
            self._realcugan_object.process(raw_in_image, raw_out_image)
        else:
            self._realcugan_object.tilesize = max(image.width, image.height)
            self._realcugan_object.process_cpu(raw_in_image, raw_out_image)

        return Image.frombytes(
            image.mode,
            (
                self._realcugan_object.scale * image.width,
                self._realcugan_object.scale * image.height,
            ),
            bytes(out_bytes),
        )

    def _get_prepadding(self) -> int:
        if self._model in ("models-se", "models-nose"):
            return {2: 18, 3: 14, 4: 19}.get(self._realcugan_object.scale, 0)
        else:
            raise ValueError(f'model "{self._model}" is not supported')

    def _get_tilesize(self):
        if self._gpuid == -1:
            return 400
        else:
            heap_budget = self._realcugan_object.get_heap_budget()
            if self._realcugan_object.scale == 2:
                if heap_budget > 1300:
                    return 400
                elif heap_budget > 800:
                    return 300
                elif heap_budget > 400:
                    return 200
                elif heap_budget > 200:
                    return 100
                else:
                    return 32
            elif self._realcugan_object.scale == 3:
                if heap_budget > 3300:
                    return 400
                elif heap_budget > 1900:
                    return 300
                elif heap_budget > 950:
                    return 200
                elif heap_budget > 320:
                    return 100
                else:
                    return 32
            elif self._realcugan_object.scale == 4:
                if heap_budget > 1690:
                    return 400
                elif heap_budget > 980:
                    return 300
                elif heap_budget > 530:
                    return 200
                elif heap_budget > 240:
                    return 100
                else:
                    return 32


class RealCUGAN(Realcugan):
    ...
