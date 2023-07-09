import sys

import cv2
import torch
from tqdm import trange

from modules.diffusion import Diffusion


def main():
    image = cv2.imread(sys.argv[1]) / 255.0

    image = torch.from_numpy(image).float()[None] * 2 - 1

    diffusion = Diffusion()

    for i in trange(1000):
        noised_image, noise = diffusion.noise_images(image, torch.LongTensor([i]))
        cv2.imshow("NoisedImage", noised_image[0].numpy() * 0.5 + 0.5)
        cv2.waitKey()


if __name__ == '__main__':
    main()
