import glob
import os
import random

import albumentations.pytorch
import albumentations as al
import cv2
import numpy as np
import torch.utils.data
import torch
import torchvision.datasets


class ImagesDataset(torch.utils.data.Dataset):
    def __init__(self, images_paths, image_transform=None):
        assert len(images_paths) > 0
        self.images_paths = images_paths

        self.image_transform = image_transform

    def __len__(self):
        return len(self.images_paths)

    def __getitem__(self, item):
        image = cv2.imread(self.images_paths[item])
        image = np.divide(image, 255.0, dtype=np.float32)

        if self.image_transform is not None:
            image = self.image_transform(image=image)

        image = torch.from_numpy(image).permute(2, 0, 1)

        return image


class AlbumentationsWrapper:
    def __init__(self, image_size):
        self.transforms = al.Sequential([
            al.Resize(image_size[1], image_size[0]),
            al.HorizontalFlip(),
            al.pytorch.ToTensorV2()
        ])

    def __call__(self, image):
        return self.transforms(image=image)["image"] / 127.5 - 1


def build_dataset(images_folder_path, image_size):
    images_dataset = torchvision.datasets.ImageFolder(
        images_folder_path, transform=AlbumentationsWrapper(image_size), loader=cv2.imread
    )
    return images_dataset


def main():
    dataset = build_dataset(r"H:\cifar10_64\test", (64, 64))
    image, classes = dataset[random.randint(0, len(dataset))]
    print(classes)
    cv2.imshow("Image", image.permute(1, 2, 0).numpy() * 0.5 + 0.5)
    cv2.waitKey(0)



if __name__ == '__main__':
    main()
