import torch
from model import DDColor
import cv2
import numpy as np
import os
from tqdm import tqdm


INPUT_DIR = "test2017/"
OUTPUT_DIR = "output_test2017/"

def inference(model, image, device):
    model.to(device)
    image = image.to(device)
    model.eval()
    with torch.no_grad():
        output = model(image)
    return output

def infer_one_image(model, image_path, output_path, device):
    img = cv2.imread(image_path)
    img = (img / 255.0).astype(np.float32)
    img = cv2.resize(img, (256, 256))
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    img_ab = img_lab[:, :, 1:]
    img_l = img_lab[:, :, :1]
    img_gray_lab = np.concatenate((img_l, np.zeros_like(img_l), np.zeros_like(img_l)), axis=-1)
    img_gray_rgb = cv2.cvtColor(img_gray_lab, cv2.COLOR_Lab2RGB)

    tensor_gray_rgb = torch.from_numpy(img_gray_rgb.transpose((2, 0, 1))).float()

    out_ab_batch = inference(model, tensor_gray_rgb.unsqueeze(0), device)
    out_ab = out_ab_batch[0].cpu().numpy().transpose((1, 2, 0))

    out_lab = np.concatenate((img_l, out_ab), axis=-1)

    out_bgr = cv2.cvtColor(out_lab, cv2.COLOR_Lab2BGR)
    out_bgr_uint8 = (out_bgr * 255.0).astype(np.uint8)
    cv2.imwrite(output_path, out_bgr_uint8)

if __name__ == '__main__':
    MAX_IMAGES = 10000

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DDColor(num_queries=100, num_scales=3, nf=512, num_output_channels=2)
    model.load_state_dict(torch.load("checkpoints/ddcolor_epoch20.pth")["generator_state_dict"])

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    loop = tqdm(os.listdir(INPUT_DIR)[:MAX_IMAGES], leave=True)
    for filename in loop:
        infer_one_image(model, INPUT_DIR + filename, OUTPUT_DIR + filename, device)


