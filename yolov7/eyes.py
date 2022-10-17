import time

import cv2
import keyboard
import numpy as np
import pyautogui as py2
import win32api
import win32con

import torch

from yolov7.models.experimental import attempt_load
from yolov7.utils.datasets import LoadImages
from yolov7.utils.general import check_img_size, non_max_suppression, set_logging
from yolov7.utils.torch_utils import select_device


def click(x, y):
    win32api.SetCursorPos((x, y))
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0)


def capture_screen():
    image = py2.screenshot()
    image.save(r"E:\WORK\PetPalsEyes\image\eyes.png")

    white = py2.screenshot(region=(500, 830, 5, 5))

    return r"E:\WORK\PetPalsEyes\image\eyes.png", white


def detect_circle(image, imgsz, stride):
    dataset = LoadImages(image, img_size=imgsz, stride=stride)

    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

    with torch.no_grad():
        pred = model(img)[0]

    pred = non_max_suppression(pred, 0.6)

    x, y = torch.ceil(pred[0][0][:2]) * 3

    print(f"X : {x}, Y : {y}")

    return int(x) + 25, int(y) + 25


# <--------------------------------------------------->

set_logging()
device = select_device('cpu')
half = device.type != 'cpu'

model = attempt_load("./weights/yolov7-petpals.pt", map_location=device)
stride = int(model.stride.max())
img_size = check_img_size(640, s=stride)

# <--------------------------------------------------->

for i in range(5):
    print("Bot start in {}".format(5 - i))
    time.sleep(1)

while keyboard.is_pressed('q') == False:
    img, white = capture_screen()

    r, g, b = white.getpixel((2, 2))

    w_x, w_y = 550, 830

    if r == 255 and g == 255 and b == 255:

        click(w_x, w_y)
        time.sleep(1)

        try:
            x, y = detect_circle(img, img_size, stride)
            if x is not None:
                click(x, y)
                time.sleep(0.6)
        except:
            print("No Circle")
            time.sleep(0.1)


    else:

        try:
            x, y = detect_circle(img, img_size, stride)
            if x is not None:
                click(x, y)
                time.sleep(0.6)
        except:
            print("No Circle")
            time.sleep(0.1)