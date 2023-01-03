import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from yolov7.models.experimental import attempt_load
from yolov7.utils.datasets import LoadStreams, LoadImages
from yolov7.utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from yolov7.utils.plots import plot_one_box
from yolov7.utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

img = "image/eyes2.png"
weights = "yolov7/weights/yolov7-petpals.pt"
imgsz = 640
conf = 0.6

# Initialize
set_logging()
device = select_device('0')
half = device.type != 'cpu'  # half precision only supported on CUDA

# Load model
model = attempt_load(weights, map_location=device)  # load FP32 model
stride = int(model.stride.max())  # model stride
imgsz = check_img_size(imgsz, s=stride)  # check img_size

if half:
    model.half()  # to FP16

dataset = LoadImages(img, img_size=imgsz, stride=stride)

# Run inference
if device.type != 'cpu':
    model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once

old_img_w = old_img_h = imgsz
old_img_b = 1

for path, img, im0s, vid_cap in dataset:
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

# Inference
with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
    pred = model(img)[0]

# Apply NMS
pred = non_max_suppression(pred, 0.6)

x, y = torch.ceil(pred[0][0][:2]) * 3

print(x, y)