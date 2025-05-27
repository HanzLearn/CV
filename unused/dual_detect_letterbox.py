import os
import sys
import argparse
import glob
import time
import cv2
import numpy as np
from ultralytics import YOLO

# Letterbox function to resize and pad the image
def letterbox(image, new_shape=(640, 640), color=(114, 114, 114)):
    """Resize and pad image while keeping aspect ratio."""
    h, w = image.shape[:2]
    scale = min(new_shape[0] / h, new_shape[1] / w)
    nh, nw = int(h * scale), int(w * scale)
    image_resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)

    # Compute padding
    pad_w = new_shape[1] - nw
    pad_h = new_shape[0] - nh
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left

    # Pad image to get exact new_shape
    image_padded = cv2.copyMakeBorder(image_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return image_padded

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

# Print class names and indices
print("\nRack Model Classes:")
for cls_id, name in rack_labels.items():
    print(f"{cls_id}: {name}")

print("\nObject Model Classes:")
for cls_id, name in obj_labels.items():
    print(f"{cls_id}: {name}")

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

bbox_colors = [(164,120,87), (68,148,228), (93,97,209), (178,182,133), (88,159,106), 
              (96,202,231), (159,124,168), (169,162,241), (98,118,150), (172,176,184)]
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
        if not ret:
            break

    # Resize and pad the frame to 640x640 (letterboxing)
    frame = letterbox(frame, (640, 640))

    # RACK DETECTION
    rack_results = rack_model(frame, verbose=False)
    rack_boxes = rack_results[0].boxes
    rack_y_ranges = []
    # print(f'Here : {rack_boxes}')

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
    obj_data = []
    print(f"Here Obj : {obj_boxes}") # !! Testing

    

    # Iterate over each object box

    # First, collect all object data
    for box in obj_boxes:
        xyxy = box.xyxy.cpu().numpy().squeeze()
        xmin, ymin, xmax, ymax = map(int, xyxy)
        cx, cy = (xmin + xmax) // 2, (ymin + ymax) // 2
        cls_id = int(box.cls.item())
        conf = box.conf.item()
        if conf < min_thresh:
            continue
        obj_data.append((cx, cy, xmin, ymin, xmax, ymax, cls_id, conf))

    # Sort by centroid X (left to right). Use reverse=True for right to left.
    obj_data.sort(key=lambda x: (x[1], -x[0]))

    # Dictionary to hold objects per rack index
    rack_objects = {i: [] for i in range(len(rack_data))}

    # Step 1: Assign each object to the correct rack based on cy
    for obj in obj_data:
        cx, cy = obj[0], obj[1]
        for i, (r_ymin, r_ymax, _, _) in enumerate(rack_data):
            if r_ymin <= cy <= r_ymax:
                rack_objects[i].append(obj)
                break  # Assigned to one rack only

    # Step 2: Sort each rack's object list by cx (left to right)
    for i in rack_objects:
        rack_objects[i].sort(key=lambda x: x[0])  # x[0] is cx

    sorted_obj_data = []
    for i in sorted(rack_objects):  # From top to bottom racks
        sorted_obj_data.extend(rack_objects[i])

    # Now draw in sorted order
    for cx, cy, xmin, ymin, xmax, ymax, cls_id, conf in sorted_obj_data:
        color = bbox_colors[cls_id % len(bbox_colors)]
        label = f"{obj_labels[cls_id]}"

        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
        labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        label_ymin = max(ymin, labelSize[1] + 10)
        cv2.rectangle(
            frame,
            (xmin, label_ymin - labelSize[1] - 10),
            (xmin + labelSize[0], label_ymin + baseLine - 10),
            color,
            cv2.FILLED,
        )
        cv2.putText(frame, label, (xmin, label_ymin - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        obj_count += 1

    for i, (cx, cy, xmin, ymin, xmax, ymax, cls_id, conf) in enumerate(obj_data):
        print(f"[{i}] cx: {cx}, cy: {cy}, xmin: {xmin}, ymin: {ymin}, xmax: {xmax}, ymax: {ymax}, cls_id: {cls_id}, name:{obj_labels[cls_id]} , conf: {conf:.3f}")
    #!! Testing

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
