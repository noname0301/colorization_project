import torch
from model import DDColor
import cv2
import numpy as np
import os

def inference(model, image):
    model.eval()
    with torch.no_grad():
        output = model(image)
    return output

if __name__ == '__main__':
    device = torch.device("cpu")
    model = DDColor(num_queries=100, num_scales=3, nf=512, num_output_channels=2).to(device)
    model.load_state_dict(torch.load("checkpoints/ddcolor_epoch20.pth")["generator_state_dict"])
    root_dir = "val_input_test/"
    output_dir = "val_output_test/"

    os.makedirs(output_dir, exist_ok=True)
    for filename in os.listdir(root_dir):
        img = cv2.imread(root_dir + filename)
        img = (img / 255.0).astype(np.float32)
        img = cv2.resize(img, (256, 256))
        img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
        img_ab = img_lab[:, :, 1:]
        img_l = img_lab[:, :, :1]
        img_gray_lab = np.concatenate((img_l, np.zeros_like(img_l), np.zeros_like(img_l)), axis=-1)
        img_gray_rgb = cv2.cvtColor(img_gray_lab, cv2.COLOR_Lab2RGB)

        tensor_gray_rgb = torch.from_numpy(img_gray_rgb.transpose((2, 0, 1))).float()

        out_ab_batch = inference(model, tensor_gray_rgb.unsqueeze(0).to(device))
        out_ab = out_ab_batch[0].cpu().numpy().transpose((1, 2, 0))

        out_lab = np.concatenate((img_l, out_ab), axis=-1)

        out_bgr = cv2.cvtColor(out_lab, cv2.COLOR_Lab2BGR)
        out_bgr_uint8 = (out_bgr * 255.0).astype(np.uint8)
        cv2.imwrite(output_dir + filename, out_bgr_uint8)
