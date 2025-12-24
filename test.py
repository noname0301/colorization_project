import cv2
import torch
import kornia.color as K
import matplotlib.pyplot as plt

# Load image with OpenCV (BGR, uint8)
img_bgr = cv2.imread('val2017\\000000000139.jpg')

# Convert BGR → RGB
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# Convert to float tensor in [0,1] and permute to CxHxW
img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0  # 3xHxW

# Add batch dim for kornia
img_tensor = img_tensor.unsqueeze(0)  # 1x3xHxW

# Convert RGB → LAB using Kornia
lab_tensor = K.rgb_to_lab(img_tensor)  # 1x3xHxW

# Extract L channel
L = lab_tensor[:, 0:1, :, :]  # 1x1xHxW

# Squeeze to 2D for visualization
L_img = L.squeeze(0).squeeze(0)  # HxW

# LAB L channel range is [0, 100], scale to [0,1] for matplotlib
L_img = L_img / 100.0

# Convert to numpy
L_img = L_img.cpu().numpy()

# Display
plt.imshow(L_img, cmap='gray')
plt.axis('off')
plt.show()

