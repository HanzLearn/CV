import os
import sys
import argparse
import glob
import time
import csv
import cv2
import numpy as np
from collections import defaultdict, Counter, deque
from ultralytics import YOLO

# Create the output directory for CSV files
output_dir = r'received_data\CSV'
os.makedirs(output_dir, exist_ok=True)

# Function to resize and pad the image (letterboxing) while maintaining aspect ratio
def letterbox(image, new_shape=(640, 640), color=(114, 114, 114)):
    h, w = image.shape[:2]
    scale = min(new_shape[0] / h, new_shape[1] / w)
    nh, nw = int(h * scale), int(w * scale)
    image_resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)

    pad_w = new_shape[1] - nw
    pad_h = new_shape[0] - nh
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left

    image_padded = cv2.copyMakeBorder(image_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return image_padded

def get_unique_filename(base_name, extension, directory):
    filename = os.path.join(directory, f"{base_name}.{extension}")
    count = 1
    while os.path.exists(filename):
        filename = os.path.join(directory, f"{base_name}_{count}.{extension}")
        count += 1
    return filename

# Usage
csv_file = get_unique_filename('detection_results', 'csv', output_dir)

print(f"Current Working Directory: {os.getcwd()}")
print(f"Full CSV Path: {os.path.abspath(csv_file)}")

# Set up command-line argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--rack_model', help='Path to YOLO rack model file', required=True)
parser.add_argument('--obj_model', help='Path to YOLO object model file', required=True)
parser.add_argument('--source', help='Image source: image file, folder, video file, or USB index (e.g., usb0)', required=True)
parser.add_argument('--thresh', help='Minimum confidence threshold', default=0.5, type=float)
parser.add_argument('--resolution', help='Resolution WxH (e.g., 640x480)', default=None)
parser.add_argument('--record', help='Record results to demo1.avi', action='store_true')
args = parser.parse_args()

# Assign variables based on the arguments
rack_model_path = args.rack_model
obj_model_path = args.obj_model
img_source = args.source
min_thresh = args.thresh
user_res = args.resolution
record = args.record

# Load YOLO models for rack and object detection
rack_model = YOLO(rack_model_path, task='detect')
obj_model = YOLO(obj_model_path, task='detect')
rack_labels = rack_model.names
obj_labels = obj_model.names

# Print available class labels for both models
print("\nRack Model Classes:")
for cls_id, name in rack_labels.items():
    print(f"{cls_id}: {name}")

print("\nObject Model Classes:")
for cls_id, name in obj_labels.items():
    print(f"{cls_id}: {name}")

# Define valid image and video extensions
img_ext = ['.jpg', '.jpeg', '.png', '.bmp']
vid_ext = ['.avi', '.mov', '.mp4', '.mkv', '.wmv']

# Determine the source type (image, folder, video, or USB)
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

# Check if resolution was provided by the user
resize = False
if user_res:
    resize = True
    resW, resH = map(int, user_res.split('x'))

# Set up video recording if required
if record:
    if source_type not in ['video','usb'] or not user_res:
        print('Recording only valid for video/usb with resolution'); sys.exit()
    recorder = cv2.VideoWriter('demo1.avi', cv2.VideoWriter_fourcc(*'MJPG'), 30, (resW, resH))

# Handle image, folder, video, or USB input
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

# Predefined colors for bounding boxes
bbox_colors = [(164,120,87), (68,148,228), (93,97,209), (178,182,133), (88,159,106), 
              (96,202,231), (159,124,168), (169,162,241), (98,118,150), (172,176,184)]

# Variables for FPS tracking and frame handling
avg_fps = 0
fps_buffer = []
fps_len = 100
img_count = 0
frame_idx = 0
csv_data = []

# Track history for each object
track_class_history = defaultdict(lambda: deque(maxlen=30))

# Maintain persistent memory of detected items per rack (to build a planogram-style structure)
rack_object_memory = []

# Start processing frames
while True:
    t_start = time.perf_counter()

    # Read a frame based on the source type
    if source_type in ['image', 'folder']:
        if img_count >= len(imgs_list): break
        frame = cv2.imread(imgs_list[img_count]); img_count += 1
    else:
        ret, frame = cap.read()
        if not ret:
            break

    frame_id_str = f"frame_{frame_idx}"
    frame_idx += 1

    # Apply letterbox resize and padding to maintain aspect ratio
    frame = letterbox(frame, (640, 640))

    # Run rack model to get bounding boxes and other information
    rack_results = rack_model.track(frame, persist=True)
    rack_boxes = rack_results[0].boxes
    rack_data = []

    for box in rack_boxes:
        xyxy = box.xyxy.cpu().numpy().squeeze()
        ymin, ymax = int(xyxy[1]), int(xyxy[3])
        xmin, xmax = int(xyxy[0]), int(xyxy[2])
        rack_data.append((ymin, ymax, xmin, xmax))

    rack_data.sort(key=lambda x: x[0])

    # Initialize memory per rack (once we know how many racks there are)
    if not rack_object_memory:
        rack_object_memory = [set() for _ in range(len(rack_data))]

    # Draw bounding boxes and labels for rack detections
    for i, (ymin, ymax, xmin, xmax) in enumerate(rack_data):
        label = f"Rack {i+1}"
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 255), 2)
        cv2.putText(frame, label, (xmin, ymin-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)

    # Run object model to get object detections
    obj_results = obj_model.track(frame, persist=True)
    obj_boxes = obj_results[0].boxes
    obj_data = []

    # Process object bounding boxes and track IDs
    for box in obj_boxes:
        xyxy = box.xyxy.cpu().numpy().squeeze()
        xmin, ymin, xmax, ymax = map(int, xyxy)
        cx, cy = (xmin + xmax) // 2, (ymin + ymax) // 2
        original_cls_id = int(box.cls.item())
        conf = box.conf.item()
        if conf < min_thresh:
            continue
        track_id = int(box.id.item()) if box.id is not None else -1

        # Update tracked object class based on history (most common class)
        if track_id != -1:
            track_class_history[track_id].append(original_cls_id)
            cls_id = Counter(track_class_history[track_id]).most_common(1)[0][0]
        else:
            cls_id = original_cls_id

        obj_data.append((cx, cy, xmin, ymin, xmax, ymax, cls_id, conf, track_id))

    # Sort detected objects into racks based on vertical position
    racks_objs = [[] for _ in range(len(rack_data))]

    for obj in obj_data:
        cx, cy, xmin, ymin, xmax, ymax, cls_id, conf, track_id = obj
        for rack_idx, (rack_ymin, rack_ymax, _, _) in enumerate(rack_data):
            if rack_ymin <= cy <= rack_ymax:
                racks_objs[rack_idx].append(obj)
                break

    # Sort objects in each rack by their X coordinate
    for rack_objs in racks_objs:
        rack_objs.sort(key=lambda x: x[0], reverse=True)

    # Display info and handle CSV logging with planogram-style memory
    obj_count = 0
    for rack_idx, rack_objs in enumerate(racks_objs):
        print(f"\nRack {rack_idx + 1}:")
        for i, (cx, cy, xmin, ymin, xmax, ymax, cls_id, conf, track_id) in enumerate(rack_objs):
            print(f"  [{i}] cx: {cx}, cy: {cy}, xmin: {xmin}, ymin: {ymin}, xmax: {xmax}, ymax: {ymax}, cls_id: {cls_id}, name: {obj_labels[cls_id]} , conf: {conf:.3f} , track_id: {track_id}")

            color = bbox_colors[cls_id % len(bbox_colors)]
            label = f"{track_id}"
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

            # Planogram-style unique object check
            object_key = (cls_id, cx, cy)
            if object_key not in rack_object_memory[rack_idx]:
                rack_object_memory[rack_idx].add(object_key)
                csv_data.append({
                    "frame_id": frame_id_str,
                    "rack_id": f"Rack_{rack_idx+1}",
                    "track_id": track_id,
                    "class_id": cls_id,
                    "class_name": obj_labels[cls_id],
                    "confidence": conf,
                    "xmin": xmin,
                    "ymin": ymin,
                    "xmax": xmax,
                    "ymax": ymax,
                    "cx": cx,
                    "cy": cy
                })

    # FPS and display
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

# Release
if source_type in ['video', 'usb']: 
    cap.release()
if record: 
    recorder.release()
cv2.destroyAllWindows()

# Write the CSV
if csv_data:
    with open(csv_file, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=csv_data[0].keys())
        writer.writeheader()
        writer.writerows(csv_data)

# Print most common classes per track ID
print("\nMost Common Classes by Track ID:")
for track_id in sorted(track_class_history.keys()):
    history = track_class_history[track_id]
    most_common = Counter(history).most_common(1)[0]
    class_id = most_common[0]
    class_name = obj_labels.get(class_id, f"Unknown_{class_id}")
    print(f"  Track {track_id}: {class_name} (Class ID: {class_id}) | Count: {most_common[1]}/{len(history)}")

print(f"Results saved to: {csv_file}")


