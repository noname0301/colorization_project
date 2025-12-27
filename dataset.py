import os
import cv2
import torch
from torch.utils.data import Dataset
import numpy as np


class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.filenames = sorted(os.listdir(self.root_dir))
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        name = self.filenames[idx]

        img = cv2.imread(os.path.join(self.root_dir, name))
        img = (img / 255.0).astype(np.float32)
        img_resized = cv2.resize(img, (256, 256))

        img_lab = cv2.cvtColor(img_resized, cv2.COLOR_BGR2Lab)
        img_l = img_lab[:, :, :1]
        img_ab = img_lab[:, :, 1:]

        img_gray_lab = np.concatenate((img_l, np.zeros_like(img_l), np.zeros_like(img_l)), axis=-1)
        img_gray_rgb = cv2.cvtColor(img_gray_lab, cv2.COLOR_LAB2RGB)

        tensor_gray_rgb = torch.from_numpy(img_gray_rgb.transpose((2, 0, 1))).float()
        tensor_ab = torch.from_numpy(img_ab.transpose((2, 0, 1))).float()

        tensor_l = torch.from_numpy(img_l.transpose((2, 0, 1))).float()
        tensor_rgb = torch.from_numpy(img_resized.transpose((2, 0, 1))).float()

        return tensor_gray_rgb, tensor_rgb, tensor_l, tensor_ab
    

if __name__ == '__main__':
    dataset = ImageDataset('train2017/')
    print(len(dataset))


