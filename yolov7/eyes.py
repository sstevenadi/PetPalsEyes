import time

import keyboard
import pyautogui as py2
import torch
import win32api
import win32con

from models.experimental import attempt_load
from utils.datasets import LoadImages
from utils.general import check_img_size, non_max_suppression, set_logging
from utils.torch_utils import select_device

def click(x, y):
    win32api.SetCursorPos((x, y))
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0)


def capture_screen():
    image = py2.screenshot()
    image.save(r"D:\WORK\PetPalsEyes\image\eyes.png")

    white = py2.screenshot(region=(495, 865, 5, 5))

    return r"D:\WORK\PetPalsEyes\image\eyes.png", white


def detect_circle(image, imgsz, stride):
    # old_img_w = old_img_h = img_size
    # old_img_b = 1

    dataset = LoadImages(image, img_size=imgsz, stride=stride)

    print("dataset")

    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        print("ok")

        # if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
        #     old_img_b = img.shape[0]
        #     old_img_h = img.shape[2]
        #     old_img_w = img.shape[3]
        #     for i in range(3):
        #         model(img)[0]

        with torch.no_grad():
            pred = model(img)[0]

        pred = non_max_suppression(pred, 0.5)

        x, y = torch.ceil(pred[0][0][:2]) * 3

        print(f"X : {x}, Y : {y}")

        return int(x) + 25, int(y) + 25

if __name__ == '__main__':

    # <--------------------------------------------------->

    set_logging()
    device = select_device('0')
    half = device.type != 'cpu'

    model = attempt_load("./weights/yolov7_eyes2.pt", map_location=device)
    stride = int(model.stride.max())
    img_size = check_img_size(640, s=stride)

    if half:
        model.half()

    if device.type != 'cpu':
        model(torch.zeros(1, 3, img_size, img_size).to(device).type_as(next(model.parameters())))

    # <--------------------------------------------------->

    for i in range(5):
        print("Bot start in {}".format(5 - i))
        time.sleep(1)

    while not keyboard.is_pressed('q'):
        img, white = capture_screen()

        r, g, b = white.getpixel((2, 2))

        w_x, w_y = 570, 860

        if r == 255 and g == 255 and b == 255:

            click(w_x, w_y)
            time.sleep(0.3)

            try:
                x, y = detect_circle(img, img_size, stride)
                if x is not None:
                    click(x, y)
                    time.sleep(0.1)
            except:
                print("No Circle")
                time.sleep(0.1)

        else:

            try:
                x, y = detect_circle(img, img_size, stride)
                if x is not None:
                    click(x, y)
                    time.sleep(0.1)
            except:
                print("No Circle")
                time.sleep(0.1)
