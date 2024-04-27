from ultralytics import YOLO
import numpy as np
import cv2
import cvzone
import math
import time
import torch
import math
from ultralytics.solutions import heatmap


model = YOLO('yolov8s-world.pt')
model.to('cuda')

classNames = ["person", "table"]

# add classes if needed: like `Trump`` if he's visiting Aaltoes or `logo`, `poster`
class_colors = {
    0: (255, 0, 0),  # Red color for class 0
    1: (0, 255, 0),  # Green color for class 1
    2: (0, 0, 255)   # Blue color for class 2
}
model.set_classes(classNames)

# tracker = Tracker()
cap = cv2.VideoCapture('posti.mov')
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
out = cv2.VideoWriter('output_posti.mp4', fourcc, fps, (int(cap.get(3)), int(cap.get(4))) )

heatmap_obj = heatmap.Heatmap()
heatmap_obj.set_args(colormap=cv2.COLORMAP_PARULA,
                     imw=w,
                     imh=h,
                     view_img=True,
                     shape="circle",
                     classes_names=model.names)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    results = model.track(frame, persist=True, show=False, device='cuda')
    frame = heatmap_obj.generate_heatmap(frame, results)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            
            cls_ = box.cls[0]
            conf = math.ceil((box.conf[0] * 100)) / 100
            color = class_colors.get(int(cls_), (255, 255, 255))
            name = classNames[int(cls_)]
            cvzone.cornerRect(frame, (x1, y1, w, h), colorR=color, colorC=color)
            cvzone.putTextRect(frame, f'{name} {conf}', (max(0, x1), max(35, y1)), colorB=color, thickness=3, scale=2, colorT=color)  # Add text with class name 

    out.write(frame)

cap.release()
out.release()