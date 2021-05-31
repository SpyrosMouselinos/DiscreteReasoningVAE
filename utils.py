import os

import numpy as np
import torch
from skimage.color import rgba2rgb
from skimage.io import imread
from skimage.transform import resize as imresize
from torch.utils.data import Dataset


def data_loader(fn):
    """
    Decorator to handle the deprecation of data_loader from 0.7
    :param fn: User defined data loader function
    :return: A wrapper for the data_loader function
    """

    def func_wrapper(self):
        return fn(self)

    return func_wrapper


class ClevrFolderDataset(Dataset):
    def __init__(self, folder, split='train', max_images=10_000, transform=None, mock_label=True):
        self.folder = folder
        self.split = split
        self.max_images = max_images
        self.transform = transform
        self.mock_label = mock_label
        self.idx2img = {}
        self.discover_images_and_make_correspondence_dict()

    def discover_images_and_make_correspondence_dict(self):
        self.idx2img = {i: f for i, f in enumerate(os.listdir(os.path.join(self.folder, self.split))) if '.png' in f}
        return

    def __len__(self):
        return self.max_images

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img = imread(os.path.join(self.folder, self.split, self.idx2img[idx]))
        img = rgba2rgb(img)
        img = imresize(img, (64, 64))
        img = img.astype('float32')
        img = img.transpose(2, 0, 1)[None]
        mean = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
        std = np.array([0.229, 0.224, 0.224]).reshape(1, 3, 1, 1)
        img = (img - mean) / std
        img = img[0]
        img = torch.FloatTensor(img)
        if self.mock_label:
            return img, torch.LongTensor([0])
        else:
            return img
