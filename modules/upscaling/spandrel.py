from __future__ import annotations

import dataclasses
import logging
from pathlib import Path
from typing import Iterable

import torch

from modules import devices, errors, modelloader, shared
from modules.shared import cmd_opts, opts
from modules.torch_utils import get_param
from modules.upscaler import Upscaler, UpscalerData
from modules.model_info import ModelInfo
from modules.upscaler_utils import upscale_with_model

ALLOWED_EXTENSIONS = {".pt", ".pth", ".ckpt", ".safetensors"}

DEFAULT_ESRGAN_MODELS = {
    "ESRGAN_4x": {
        "url": "https://github.com/cszn/KAIR/releases/download/v1.0/ESRGAN.pth",
        "local_name": "ESRGAN_4x.pth",
    },
}

DEFAULT_SWINIR_MODELS = {
    "SwinIR 4x": {
        "url": "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth",
        "local_name": "SwinIR_4x.pth",
    },
}

DEFAULT_REALESRGAN_MODEL_URLS = {
    "R-ESRGAN General 4xV3": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth",
    "R-ESRGAN General WDN 4xV3": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth",
    "R-ESRGAN AnimeVideo": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth",
    "R-ESRGAN 4x+": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
    "R-ESRGAN 4x+ Anime6B": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
    "R-ESRGAN 2x+": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
}

DEFAULT_SCUNET_MODELS = {
    "ScuNET GAN": {
        "url": "https://github.com/cszn/KAIR/releases/download/v1.0/scunet_color_real_gan.pth",
        "local_name": "ScuNET.pth",  # Legacy...
    },
    "ScuNET PSNR": {
        "url": "https://github.com/cszn/KAIR/releases/download/v1.0/scunet_color_real_psnr.pth",
    },
}

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class SpandrelUpscalerData(UpscalerData):
    upscaler: UpscalerSpandrel
    name: str  # human-readable name
    type: str  # ESRGAN, RealESRGAN, HAT, etc.
    model_info: ModelInfo

    @classmethod
    def from_model_info(
        cls, upscaler: UpscalerSpandrel, model_info: ModelInfo, type: str
    ):
        return cls(
            upscaler=upscaler,
            name=model_info.name,
            model_info=model_info,
            type=type,
        )

    @property
    def scale(self) -> int | None:
        if self.type == "ScuNET":
            # ScuNET is a denoiser, not an upscaler.
            return 1
        # We can't truly know this before we've loaded the model, but we can infer something
        # from the model name.
        if "4x" in self.name or "x4" in self.name:
            return 4
        if "2x" in self.name or "x2" in self.name:
            return 2
        return None


def from_model_directory(upscaler: UpscalerSpandrel, directory: Path, *, type: str):
    for model_full_filename in shared.walk_files(
        directory, allowed_extensions=ALLOWED_EXTENSIONS
    ):
        model_path = Path(model_full_filename)
        yield SpandrelUpscalerData.from_model_info(
            upscaler,
            ModelInfo(
                name=model_path.stem,
                local_name=model_path.name,
                local_directory=model_path.parent,
            ),
            type=type,
        )


def from_default_infos_dict(
    upscaler, infos_dict: dict[str, dict | str], directory: Path, *, type: str
):
    for name, info in infos_dict.items():
        if isinstance(info, str):  # Just an URL
            info = {"url": info}
        model_info = ModelInfo(name=name, local_directory=directory, **info)
        yield SpandrelUpscalerData.from_model_info(upscaler, model_info, type=type)


def get_esrgan_models(upscaler: UpscalerSpandrel):
    model_directory = Path(cmd_opts.esrgan_models_path)
    yield from from_default_infos_dict(
        upscaler, DEFAULT_ESRGAN_MODELS, model_directory, type="ESRGAN"
    )
    yield from from_model_directory(upscaler, model_directory, type="ESRGAN")


def get_realesrgan_models(upscaler: UpscalerSpandrel):
    model_directory = Path(cmd_opts.realesrgan_models_path)
    yield from from_default_infos_dict(
        upscaler, DEFAULT_REALESRGAN_MODEL_URLS, model_directory, type="RealESRGAN"
    )
    yield from from_model_directory(upscaler, model_directory, type="RealESRGAN")


def get_swinir_models(upscaler: UpscalerSpandrel):
    model_directory = Path(cmd_opts.swinir_models_path)
    yield from from_default_infos_dict(
        upscaler, DEFAULT_SWINIR_MODELS, model_directory, type="SwinIR"
    )
    yield from from_model_directory(upscaler, model_directory, type="SwinIR")


def get_scunet_models(upscaler: UpscalerSpandrel):
    model_directory = Path(cmd_opts.scunet_models_path)
    yield from from_default_infos_dict(
        upscaler, DEFAULT_SCUNET_MODELS, model_directory, type="ScuNET"
    )
    yield from from_model_directory(upscaler, model_directory, type="ScuNET")


class UpscalerSpandrel(Upscaler):
    name = "Spandrel"

    def discover(self) -> Iterable[SpandrelUpscalerData]:
        yield from get_esrgan_models(self)
        yield from get_realesrgan_models(self)
        yield from get_swinir_models(self)
        yield from get_scunet_models(self)
        yield from from_model_directory(
            self, Path(shared.models_path) / "HAT", type="HAT"
        )

    def do_upscale(self, img, size, data: SpandrelUpscalerData):
        device = devices.get_device_for(data.type.lower())
        prefer_half = not cmd_opts.no_half and not cmd_opts.upcast_sampling

        model_info = data.model_info
        try:
            # TODO(akx): should we LRU cache these models?
            model_info.download_if_required()
            model_descriptor = modelloader.load_spandrel_model(
                model_info.local_path,
                device=device,
                prefer_half=prefer_half,
                expected_architecture=data.type,
            )
        except Exception:
            errors.report(f"Unable to load Spandrel model {model_info}", exc_info=True)
            return img

        should_compile = (
            False  # TODO: make this configurable for models other than SwinIR
        )
        offload_after_upscale = False  # TODO: make this configurable

        if data.type == "ESRGAN":
            tile_size = opts.ESRGAN_tile
            tile_overlap = opts.ESRGAN_tile_overlap
        elif data.type == "SCUNET":
            tile_size = opts.SCUNET_tile
            tile_overlap = opts.SCUNET_tile_overlap
        elif data.type == "SwinIR":
            tile_size = opts.SWIN_tile
            tile_overlap = opts.SWIN_tile_overlap
            should_compile = bool(getattr(opts, "SWIN_torch_compile", False))
        else:
            logger.warning("Using ESRGAN tile size for %s model", data.type)
            tile_size = opts.ESRGAN_tile
            tile_overlap = opts.ESRGAN_tile_overlap

        try:
            model_descriptor.to(device)
            if should_compile:
                try:
                    logger.info("Compiling upscaling model")
                    torch.compile(model_descriptor.model)
                except Exception:
                    logger.warning("Failed to compile upscaling model", exc_info=True)
            logger.info(
                "Starting upscale of image of size %s with %s (tile size %d, overlap %d); device %s, dtype %s",
                img.size,
                data,
                tile_size,
                tile_overlap,
                device,
                get_param(model_descriptor).dtype,
            )
            return upscale_with_model(
                model_descriptor,
                img,
                tile_size=tile_size,
                tile_overlap=tile_overlap,
                desc=f"{data.name} upscale using {device}",
            )
        finally:
            if offload_after_upscale:
                logger.debug("Offloading upscale model to CPU")
                model_descriptor.to(devices.cpu)
