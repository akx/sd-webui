from __future__ import annotations

import dataclasses
from typing import Iterable

from PIL import Image

from modules.upscaler import Upscaler, UpscalerData


@dataclasses.dataclass(frozen=True)
class PillowUpscalerData(UpscalerData):
    upscaler: PillowUpscaler
    name: str
    resampling: Image.Resampling | None = None
    priority: int


class PillowUpscaler(Upscaler):
    name = "None"

    def do_upscale(self, img, size, data: PillowUpscalerData):
        if data.resampling:
            return img.resize(size, resample=data.resampling)
        return img

    def discover(self) -> Iterable[UpscalerData]:
        return [
            PillowUpscalerData(upscaler=self, name="None", priority=-2),
            PillowUpscalerData(upscaler=self, name="Nearest", resampling=Image.Resampling.NEAREST, priority=-1),
            PillowUpscalerData(upscaler=self, name="Lanczos", resampling=Image.Resampling.LANCZOS, priority=-1),
        ]
