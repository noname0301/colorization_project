import torch
from model import DDColor
import cv2
import kornia.color as K
import numpy as np
import os

def inference(model, image):
    model.eval()
    with torch.no_grad():
        output = model(image)
    return output

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DDColor("convnext-t", num_queries=100, num_scales=3, nf=512, num_channels=2).to(device)
    model.load_state_dict(torch.load("checkpoints/ddcolor_epoch200.pth")["generator_state_dict"])
    os.makedirs("output", exist_ok=True)
    for filename in os.listdir("test_dataset/"):
        img = cv2.imread("test_dataset/" + filename)
        img = cv2.resize(img, (256, 256))
        img = torch.from_numpy(img).permute(2,0,1).float() / 255.0  # CxHxW
        img = K.rgb_to_lab(img.unsqueeze(0)).squeeze(0)
        L_single = img[0:1, :, :] / 100.0  # 1xHxW
        L = L_single.repeat(3, 1, 1) # 3xHxW to feed the 3-channel encoder
        L = L.unsqueeze(0).to(device)

        output = inference(model, L)


        output = K.lab_to_rgb(torch.cat([(L_single * 100).unsqueeze(0).to(device), output.clamp(-1,1) * 128], dim=1)).squeeze(0)  # 3xHxW
        output = output.permute(1,2,0).cpu().numpy()
        output = (output * 255).astype(np.uint8)
        output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
        cv2.imwrite("output/" + filename, output)