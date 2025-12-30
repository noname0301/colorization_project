import cv2
import os
from metrics import calculate_psnr, calculate_ssim, calculate_ms_ssim, calculate_colorfulness, calculate_fid
import numpy as np
import torch


if __name__ == '__main__':
    input_dir = "val_input_test/"
    output_dir = "val_output_test/"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    images_real = []
    images_fake = []

    for filename in os.listdir(input_dir):
        img = cv2.imread(input_dir + filename)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = (img / 255.0).astype(np.float32)
        img = cv2.resize(img, (256, 256))
        img = np.transpose(img, (2, 0, 1))
        images_real.append(img)

    for filename in os.listdir(output_dir):
        img = cv2.imread(output_dir + filename)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = (img / 255.0).astype(np.float32)
        img = cv2.resize(img, (256, 256))
        img = np.transpose(img, (2, 0, 1))
        images_fake.append(img)

    images_real = np.array(images_real)
    images_fake = np.array(images_fake)
    images_real = torch.from_numpy(images_real)
    images_fake = torch.from_numpy(images_fake)

    print(images_real.shape, images_fake.shape)

    psnr = calculate_psnr(images_real, images_fake)
    ssim = calculate_ssim(images_real, images_fake)
    ms_ssim = calculate_ms_ssim(images_real, images_fake)
    colorfulness = calculate_colorfulness(images_fake)
    fid = calculate_fid(images_real, images_fake, device=device)

    print("PSNR:", psnr)
    print("SSIM:", ssim)
    print("MS-SSIM:", ms_ssim)
    print("Colorfulness:", colorfulness)
    print("FID:", fid)
