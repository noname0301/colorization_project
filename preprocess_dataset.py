import os
import cv2

img_path = 'val2017/'
os.makedirs('dataset/train/images', exist_ok=True)

for filename in os.listdir(img_path):
    image = cv2.imread(img_path + filename)
    image = cv2.resize(image, (256, 256))
    cv2.imwrite('dataset/val/ground_truths/' + filename, image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('dataset/val/images/' + filename, image)