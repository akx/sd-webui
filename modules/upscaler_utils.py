import numpy as np
import torch
import tqdm
from PIL import Image

from modules import devices, images


def upscale_without_tiling(model, img: Image.Image):
    img = np.array(img)
    img = img[:, :, ::-1]
    img = np.ascontiguousarray(np.transpose(img, (2, 0, 1))) / 255
    img = torch.from_numpy(img).float()
    img = img.unsqueeze(0).to(devices.device_esrgan)
    with torch.no_grad():
        output = model(img)
    output = output.squeeze().float().cpu().clamp_(0, 1).numpy()
    output = 255. * np.moveaxis(output, 0, 2)
    output = output.astype(np.uint8)
    output = output[:, :, ::-1]
    return Image.fromarray(output, 'RGB')


def upscale_with_model(model, img: Image.Image, *, tile_size: int, tile_overlap: int = 0):
    if tile_size <= 0:
        return upscale_without_tiling(model, img)

    grid = images.split_grid(img, tile_size, tile_size, tile_overlap)
    newtiles = []
    scale_factor = 1

    for y, h, row in tqdm.tqdm(grid.tiles, desc="tiled upscale"):
        newrow = []
        for tiledata in row:
            x, w, tile = tiledata

            output = upscale_without_tiling(model, tile)
            scale_factor = output.width // tile.width

            newrow.append([x * scale_factor, w * scale_factor, output])
        newtiles.append([y * scale_factor, h * scale_factor, newrow])

    newgrid = images.Grid(newtiles, grid.tile_w * scale_factor, grid.tile_h * scale_factor, grid.image_w * scale_factor,
                          grid.image_h * scale_factor, grid.overlap * scale_factor)
    output = images.combine_grid(newgrid)
    return output
