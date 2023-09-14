import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

class DriveDataset(Dataset):
    def __init__(self, images_path, masks_path, scale = 0.5):

        self.images_path = images_path
        # print(len(self.images_path))
        self.masks_path = masks_path
        self.n_samples = len(images_path)
        self.scale = scale

    def __getitem__(self, index):
        """ Reading image """
        # print(index)
        # print(self.images_path[index])
        # print(self.images_path[index])
        img = cv2.imread(self.images_path[index], cv2.IMREAD_COLOR)
        w = int(img.shape[0]*self.scale)
        h = int(img.shape[1]*self.scale)
        image = cv2.resize(img, (w, h))
        image = image/255.0 ## (512, 512, 3)
        image = np.transpose(image, (2, 0, 1))  ## (3, 512, 512)
        image = image.astype(np.float32)
        image = torch.from_numpy(image)

        """ Reading mask """
        # s = self.masks_path[index]
        # mask = cv2.imread(s, cv2.IMREAD_COLOR)
        # print(self.masks_path[index])
        img = cv2.imread(self.masks_path[index], cv2.IMREAD_GRAYSCALE)
        w = int(img.shape[0] * self.scale)
        h = int(img.shape[1] * self.scale)
        mask = cv2.resize(img, (w, h))
        # mask = cv2.imread(self.masks_path[index])
        mask = mask/255.0   ## (512, 512)
        mask = np.expand_dims(mask, axis=0) ## (1, 512, 512)
        mask = mask.astype(np.float32)
        mask = torch.from_numpy(mask)

        return image, mask

    def __len__(self):
        return self.n_samples