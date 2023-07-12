import sys

import cv2
import einops
import torch

from modules.lit_modules.cfg_diffusion import CFGDiffusion


def main():
    model = CFGDiffusion.load_from_checkpoint(sys.argv[1]).cuda().eval()

    noises = torch.randn(10, 3, 64, 64).cuda()
    images = model.diffusion.sample(model.model, noises, torch.LongTensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).cuda())

    images = einops.rearrange(images, "b c h w -> h (b w) c").cpu().numpy()
    cv2.imwrite("./result.png", images * 255)


if __name__ == '__main__':
    main()
