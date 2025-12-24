import os
import cv2
import torch
from torch.utils.data import Dataset
import kornia.color as K


class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.filenames = sorted(os.listdir(self.root_dir))
        self.transform = transform

    def __len__(self):
        return min(len(self.filenames), 20000)

    def __getitem__(self, idx):
        name = self.filenames[idx]
        img = cv2.imread(os.path.join(self.root_dir, name))
        img = cv2.resize(img, (256, 256))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # convert to RGB
        img = torch.from_numpy(img).permute(2,0,1).float() / 255.0  # CxHxW

        # Convert to LAB and take L channel
        img = K.rgb_to_lab(img.unsqueeze(0)).squeeze(0)  # 3xHxW
        gt = img[1:, :, :]
        L = img[0:1, :, :]  # 1xHxW
        L = L.repeat(3, 1, 1)  # 3xHxW to feed the 3-channel encoder

        if self.transform:
            L = self.transform(L)
            gt = self.transform(gt)

        return L, gt
    

if __name__ == '__main__':
    dataset = ImageDataset('train2017/')
    print(len(dataset))
    L, gt = dataset[0]
