import cv2
import os
from metrics import calculate_psnr_ssim, calculate_colorfulness, calculate_fid
import numpy as np
import torch
import json
from tqdm import tqdm


if __name__ == '__main__':
    input_dir = "test2017/"
    output_dir = "output_test2017/"

    MAX_IMAGES = 10000

    device = "cuda" if torch.cuda.is_available() else "cpu"

    images_real = []
    images_fake = []

    loop = tqdm(os.listdir(input_dir)[:MAX_IMAGES], leave=True)
    for filename in loop:
        img = cv2.imread(input_dir + filename)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = (img / 255.0).astype(np.float32)
        img = cv2.resize(img, (256, 256))
        img = np.transpose(img, (2, 0, 1))
        images_real.append(img)

    loop = tqdm(os.listdir(output_dir)[:MAX_IMAGES], leave=True)
    for filename in loop:
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

    psnr, ssim, ms_ssim = calculate_psnr_ssim(images_real, images_fake, device=device)
    colorfulness = calculate_colorfulness(images_fake, device=device)
    fid = calculate_fid(images_real, images_fake, device=device)

    metrics = {
        "psnr": psnr,
        "ssim": ssim,
        "ms_ssim": ms_ssim,
        "colorfulness": colorfulness,
        "fid": fid
    }

    print("PSNR:", psnr)
    print("SSIM:", ssim)
    print("MS-SSIM:", ms_ssim)
    print("Colorfulness:", colorfulness)
    print("FID:", fid)

    with open("evaluation_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)