from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Iterable, Protocol

import PIL
from PIL import Image

logger = logging.getLogger(__name__)


class UpscalerData(Protocol):
    """
    Protocol for upscaler data objects.

    These bind upscaler implementations to specific models and/or additional data required.
    """

    # The upscaler implementation class.
    upscaler: Upscaler

    # Human-readable name for this upscaler. Probably the name of the model.
    name: str

    # The priority is used to sort upscalers in the UI.
    # The default priority is 0, and lower numbers are higher priority.
    priority: int = 0


class Upscaler(ABC):
    name: str | None = None  # Should be set by concrete subclasses

    @abstractmethod
    def discover(self) -> Iterable[UpscalerData]:
        """
        Discover upscaler data for this upscaler, by e.g. looking for models in the file system.

        The upscaler implementation may trust that it is the single instance of its class for
        the lifetime of a given `UpscalerData` object.
        """
        return []

    @abstractmethod
    def do_upscale(self, img: PIL.Image, size: tuple[int, int], data: UpscalerData):
        return img


def upscale(img: PIL.Image, *, scale: float, data: UpscalerData) -> PIL.Image:
    """
    Upscale `img` by `scale` using the upscaler configured by `data`.
    """
    if scale <= 1:  # Nothing to do.
        return img
    dest_w = int((img.width * scale) // 8 * 8)
    dest_h = int((img.height * scale) // 8 * 8)

    for _ in range(3):
        if img.width >= dest_w and img.height >= dest_h:
            break

        shape = (img.width, img.height)

        img = data.upscaler.do_upscale(img, (dest_w, dest_h), data)

        if shape == (img.width, img.height):
            break

    if img.width != dest_w or img.height != dest_h:
        logger.warning("Requested size %dx%d, got %dx%d; resizing.", dest_w, dest_h, img.width, img.height)
        img = img.resize((int(dest_w), int(dest_h)), resample=Image.Resampling.LANCZOS)

    return img


def load_upscalers():
    # The modules need to be imported so the classes are registered as `__subclasses__` of Upscaler.
    # Built-in extensions will presumably already have been imported.
    from modules.upscaling import pillow, spandrel  # noqa: F401

    upscaler_datas = []
    for cls in Upscaler.__subclasses__():
        if not cls.name:
            # Assume this is e.g. an abstract mix-in class.
            continue
        upscaler_datas.extend(cls().discover())
    upscaler_datas.sort(key=lambda d: (d.priority, d.name.lower()))

    from modules import shared
    shared.sd_upscalers = upscaler_datas


def find_upscaler(name_or_index: int | str) -> UpscalerData:
    """
    Find an upscaler by name or index.

    If `name_or_index` is an integer, it is interpreted as an index into the list of upscalers.
    Otherwise, it is interpreted as a case-insensitive name.
    """
    from modules import shared
    if isinstance(name_or_index, int):
        return shared.sd_upscalers[name_or_index]
    name_or_index = str(name_or_index).lower()
    for d in shared.sd_upscalers:
        if d.name.lower() == name_or_index:
            return d
    raise KeyError(f"Upscaler {name_or_index!r} not found.")
