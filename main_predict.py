import sys

import cv2
import einops
import numpy as np
import torch
from PIL import Image

from modules.lit_modules.cfg_diffusion import CFGDiffusion


def main():
    model = CFGDiffusion.load_from_checkpoint(sys.argv[1]).cuda().eval()

    noises = torch.randn(10, 3, 64, 64).cuda()
    images = model.diffusion.sample(
        model.model, noises, torch.LongTensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).cuda(),
        return_history=True
    )
    # images = torch.from_numpy(np.random.rand(150, 10, 3, 64, 64))
    images = torch.flip(images, [2])
    images = torch.clip(images, 0, 1)
    images = np.uint8(einops.rearrange(images, "n b c h w -> n h (b w) c").contiguous().cpu().numpy() * 255)
    pil_images = [
        Image.fromarray(cv2.resize(image, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)) for image in images
    ]
    pil_images = pil_images + [pil_images[-1]] * 10
    pil_images[0].save("./result.gif", save_all=True, append_images=pil_images[1:], duration=100, loop=1, )


if __name__ == '__main__':
    main()
