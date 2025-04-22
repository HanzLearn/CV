import os
import sys
import argparse
import glob
import time

import cv2
import numpy as np
from ultralytics import YOLO

# Define and parse user input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--rack_model', help='Path to YOLO rack model file', required=True)
parser.add_argument('--obj_model', help='Path to YOLO object model file', required=True)
parser.add_argument('--source', help='Image source: image file, folder, video file, or USB index (e.g., usb0)', required=True)
parser.add_argument('--thresh', help='Minimum confidence threshold', default=0.5, type=float)
parser.add_argument('--resolution', help='Resolution WxH (e.g., 640x480)', default=None)
parser.add_argument('--record', help='Record results to demo1.avi', action='store_true')
args = parser.parse_args()

# Parse inputs
rack_model_path = args.rack_model
obj_model_path = args.obj_model
img_source = args.source
min_thresh = args.thresh
user_res = args.resolution
record = args.record

# Load models
rack_model = YOLO(rack_model_path, task='detect')
obj_model = YOLO(obj_model_path, task='detect')
rack_labels = rack_model.names
obj_labels = obj_model.names

# Determine source type
img_ext = ['.jpg', '.jpeg', '.png', '.bmp']
vid_ext = ['.avi', '.mov', '.mp4', '.mkv', '.wmv']

if os.path.isdir(img_source):
    source_type = 'folder'
elif os.path.isfile(img_source):
    ext = os.path.splitext(img_source)[1]
    if ext in img_ext:
        source_type = 'image'
    elif ext in vid_ext:
        source_type = 'video'
    else:
        print('Unsupported file extension.'); sys.exit()
elif 'usb' in img_source:
    source_type = 'usb'
    usb_idx = int(img_source[3:])
else:
    print('Invalid source input.'); sys.exit()

resize = False
if user_res:
    resize = True
    resW, resH = map(int, user_res.split('x'))

if record:
    if source_type not in ['video','usb'] or not user_res:
        print('Recording only valid for video/usb with resolution'); sys.exit()
    recorder = cv2.VideoWriter('demo1.avi', cv2.VideoWriter_fourcc(*'MJPG'), 30, (resW, resH))

if source_type == 'image':
    imgs_list = [img_source]
elif source_type == 'folder':
    imgs_list = [f for f in glob.glob(img_source + '/*') if os.path.splitext(f)[1] in img_ext]
elif source_type == 'video':
    cap = cv2.VideoCapture(img_source)
elif source_type == 'usb':
    cap = cv2.VideoCapture(usb_idx)

    if user_res:
        cap.set(3, resW)
        cap.set(4, resH)

bbox_colors = [(255,0,0), (0,255,0), (0,0,255), (255,255,0), (255,0,255)]
avg_fps = 0
fps_buffer = []
fps_len = 100
img_count = 0

while True:
    t_start = time.perf_counter()

    if source_type in ['image', 'folder']:
        if img_count >= len(imgs_list): break
        frame = cv2.imread(imgs_list[img_count]); img_count += 1
    else:
        ret, frame = cap.read()
        if not ret: break

    if resize:
        frame = cv2.resize(frame, (resW, resH))

    
    # RACK DETECTION
    rack_results = rack_model(frame, verbose=False)
    rack_boxes = rack_results[0].boxes
    rack_y_ranges = []

    # Collect rack bounding boxes with their ymin values
    rack_data = []
    for box in rack_boxes:
        xyxy = box.xyxy.cpu().numpy().squeeze()
        ymin, ymax = int(xyxy[1]), int(xyxy[3])
        rack_data.append((ymin, ymax, xyxy[0], xyxy[2]))  # store ymin, ymax, xmin, xmax

    # Sort racks by ymin in ascending order (highest rack will be last)
    rack_data.sort(key=lambda x: x[0])

    # Label the racks in order (from top to bottom)
    for i, (ymin, ymax, xmin, xmax) in enumerate(rack_data):
        # Label the racks starting from "Rack 1" for the highest rack
        label = f"Rack {i+1}"
        cv2.rectangle(frame, (int(xmin), ymin), (int(xmax), ymax), (0, 255, 255), 2)
        cv2.putText(frame, label, (int(xmin), ymin-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)


    # OBJECT DETECTION
    obj_results = obj_model(frame, verbose=False)
    obj_boxes = obj_results[0].boxes
    obj_count = 0

    # Iterate over each object box
    for box in obj_boxes:
        xyxy = box.xyxy.cpu().numpy().squeeze()
        xmin, ymin, xmax, ymax = map(int, xyxy)
        cx, cy = (xmin + xmax) // 2, (ymin + ymax) // 2  # Calculate the centroid of the object
        cls_id = int(box.cls.item())
        conf = box.conf.item()
        if conf < min_thresh: continue

        # Check if the object's centroid is within the bounds of any rack (based on ymin and ymax of racks)
        in_rack = any(rack_ymin <= cy <= rack_ymax for rack_ymin, rack_ymax, _, _ in rack_data)

        # Define color and label based on whether it's inside a rack
        color = (0, 255, 0) if in_rack else (0, 0, 255)
        label = f"{obj_labels[cls_id]} ({'IN' if in_rack else 'OUT'})"
        
        # Draw bounding box and label
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
        cv2.putText(frame, label, (xmin, ymin-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        obj_count += 1

    fps = 1.0 / (time.perf_counter() - t_start)
    fps_buffer.append(fps)
    if len(fps_buffer) > fps_len:
        fps_buffer.pop(0)
    avg_fps = np.mean(fps_buffer)

    cv2.putText(frame, f"FPS: {avg_fps:.2f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    cv2.putText(frame, f"Objects: {obj_count}", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    cv2.imshow("Rack & Object Detection", frame)
    if record: recorder.write(frame)

    key = cv2.waitKey(0 if source_type in ['image', 'folder'] else 5)
    if key == ord('q'): break
    elif key == ord('s'): cv2.waitKey()
    elif key == ord('p'): cv2.imwrite('capture.png', frame)

if source_type in ['video', 'usb']: cap.release()
if record: recorder.release()
cv2.destroyAllWindows()