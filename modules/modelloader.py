from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING
from urllib.parse import urlparse

import torch

from modules import shared

if TYPE_CHECKING:
    import spandrel

logger = logging.getLogger(__name__)


def load_file_from_url(
    url: str,
    *,
    model_dir: str,
    progress: bool = True,
    file_name: str | None = None,
) -> str:
    """Download a file from `url` into `model_dir`, using the file present if possible.

    Returns the path to the downloaded file.
    """
    os.makedirs(model_dir, exist_ok=True)
    if not file_name:
        parts = urlparse(url)
        file_name = os.path.basename(parts.path)
    cached_file = os.path.abspath(os.path.join(model_dir, file_name))
    if not os.path.exists(cached_file):
        print(f'Downloading: "{url}" to {cached_file}\n')
        from torch.hub import download_url_to_file
        download_url_to_file(url, cached_file, progress=progress)
    return cached_file


def load_models(model_path: str, model_url: str = None, command_path: str = None, ext_filter=None, download_name=None, ext_blacklist=None) -> list:
    """
    A one-and done loader to try finding the desired models in specified directories.

    @param download_name: Specify to download from model_url immediately.
    @param model_url: If no other models are found, this will be downloaded on upscale.
    @param model_path: The location to store/find models in.
    @param command_path: A command-line argument to search for models in first.
    @param ext_filter: An optional list of filename extensions to filter by
    @return: A list of paths containing the desired model(s)
    """
    output = []

    try:
        places = []

        if command_path is not None and command_path != model_path:
            pretrained_path = os.path.join(command_path, 'experiments/pretrained_models')
            if os.path.exists(pretrained_path):
                print(f"Appending path: {pretrained_path}")
                places.append(pretrained_path)
            elif os.path.exists(command_path):
                places.append(command_path)

        places.append(model_path)

        for place in places:
            for full_path in shared.walk_files(place, allowed_extensions=ext_filter):
                if os.path.islink(full_path) and not os.path.exists(full_path):
                    print(f"Skipping broken symlink: {full_path}")
                    continue
                if ext_blacklist is not None and any(full_path.endswith(x) for x in ext_blacklist):
                    continue
                if full_path not in output:
                    output.append(full_path)

        if model_url is not None and len(output) == 0:
            if download_name is not None:
                output.append(load_file_from_url(model_url, model_dir=places[0], file_name=download_name))
            else:
                output.append(model_url)

    except Exception:
        pass

    return output


def load_spandrel_model(
    path: str | os.PathLike,
    *,
    device: str | torch.device | None,
    prefer_half: bool = False,
    dtype: str | torch.dtype | None = None,
    expected_architecture: str | None = None,
) -> spandrel.ModelDescriptor:
    import spandrel
    model_descriptor = spandrel.ModelLoader(device=device).load_from_file(str(path))
    if expected_architecture and model_descriptor.architecture != expected_architecture:
        logger.warning(
            f"Model {path!r} is not a {expected_architecture!r} model (got {model_descriptor.architecture!r})",
        )
    half = False
    if prefer_half:
        if model_descriptor.supports_half:
            model_descriptor.model.half()
            half = True
        else:
            logger.info("Model %s does not support half precision, ignoring --half", path)
    if dtype:
        model_descriptor.model.to(dtype=dtype)
    model_descriptor.model.eval()
    logger.debug(
        "Loaded %s from %s (device=%s, half=%s, dtype=%s)",
        model_descriptor, path, device, half, dtype,
    )
    return model_descriptor
