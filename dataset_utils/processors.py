import torch
from torchvision.io import ImageReadMode, read_image
from torchvision.transforms import CenterCrop, ConvertImageDtype, Normalize, Resize
from torchvision.transforms.functional import InterpolationMode
from torchvision import transforms
from PIL import Image
import random

FLAMINGO_MEAN = (0.481, 0.458, 0.408)
FLAMINGO_STD = (0.269, 0.261, 0.276)

class DefaultTransform:
    def __init__(self, image_size=224, min_scale=0.5, mode="train"):

        self.image_size = image_size
        self.min_scale = min_scale

        if mode == "train":
            self.image_transform = transforms.Compose([
                #transforms.Resize((image_size, image_size), interpolation=Image.BICUBIC),
                transforms.Resize((image_size, image_size),
                                  interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean=FLAMINGO_MEAN, std=FLAMINGO_STD),
            ])
        else:
            self.image_transform = transforms.Compose([
                transforms.Resize((image_size, image_size), interpolation=Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(FLAMINGO_MEAN, FLAMINGO_STD),
            ])

        self.text_transform = None

    def __call__(self, image):
        return self.image_transform(image)


class RegionTransform:
    def __init__(self, image_size=224, min_scale=0.5):

        self.image_size = image_size
        self.min_scale = min_scale

        self.image_transform = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=Image.BICUBIC),
            #transforms.RandomResizedCrop(image_size, scale=(min_scale, 1.0), interpolation=Image.BICUBIC),
            #transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(FLAMINGO_MEAN, FLAMINGO_STD),
        ])
        self.text_transform = None

    def __call__(self, image):
        return self.image_transform(image)


