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

img = "image/eyes.png"
weights = "yolov7/weights/yolov7-petpals.pt"
imgsz = 640
conf = 0.6

set_logging()
device = select_device('cpu')
half = device.type != 'cpu'

model = attempt_load(weights, map_location=device)
stride = int(model.stride.max())  # model stride
imgsz = check_img_size(imgsz, s=stride) # check img_size

dataset = LoadImages(img, img_size=imgsz, stride=stride)

# names = model.module.names if hasattr(model, 'module') else model.names
# colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

# old_img_w = old_img_h = imgsz
# old_img_b = 1

for path, img, im0s, vid_cap in dataset:
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
    pred = model(img)[0]

pred = non_max_suppression(pred, conf)

x, y = torch.ceil(pred[0][0][:2])

# for i, det in enumerate(pred):
    # p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)
    #
    # gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh


    # if len(det):
    #     # Rescale boxes from img_size to im0 size
    #     det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
    #
    #     # Print results
    #     for c in det[:, -1].unique():
    #         n = (det[:, -1] == c).sum()  # detections per class
    #         s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
    #
    #     for *xyxy, conf, cls in reversed(det):
    #         xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh