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
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
'''
Output directory for CSV file
'''
output_dir = r'received_data\CSV'
os.makedirs(output_dir, exist_ok=True) # Create output directory. No error if it already exists

'''
Function to produce unique filename for CSV
'''
def get_unique_filename(base_name, extension, directory):
    count = 1
    while True:
        filename = os.path.join(directory, f"{base_name}_{count}.{extension}")
        if not os.path.exists(filename):
            return filename
        count += 1

csv_file = get_unique_filename('detection_results', 'csv', output_dir)

'''
CLI arguments
'''

parser = argparse.ArgumentParser()
parser.add_argument('--rack_model', help='Path to YOLO rack model file', required=True)
parser.add_argument('--obj_model', help='Path to YOLO object model file', required=True)
parser.add_argument('--source', help='Image source: image file, folder, video file, or USB index (e.g., usb0)', required=True)
parser.add_argument('--thresh', help='Minimum confidence threshold', default=0.5, type=float)
parser.add_argument('--resolution', help='Resolution WxH (e.g., 640x480)', default=None)
parser.add_argument('--record', help='Record results to demo1.avi', action='store_true')
args = parser.parse_args()

'''
Variables to hold passed args
'''
rack_model_path = args.rack_model
obj_model_path = args.obj_model
img_source = args.source
min_thresh = args.thresh
user_res = args.resolution
record = args.record

'''
Load YOLO models
'''
rack_model = YOLO(rack_model_path, task='detect').to(device)
obj_model = YOLO(obj_model_path, task='detect').to(device)
rack_labels = rack_model.names
obj_labels = obj_model.names


'''
Print available list of classes for debugging
'''
# print("\nRack Model Classes:")
# for cls_id, name in rack_labels.items():
#     print(f"{cls_id}: {name}")

# print("\nObject Model Classes:")
# for cls_id, name in obj_labels.items():
#     print(f"{cls_id}: {name}")

print("Rack Model Device:", rack_model.device)
print("Object Model Device:", obj_model.device)
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())

'''
List of valid extensions for images and videos
'''
img_ext = ['.jpg', '.jpeg', '.png', '.bmp']
vid_ext = ['.avi', '.mov', '.mp4', '.mkv', '.wmv']

'''
Determine source type of input
'''
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

'''
Determine whether res provided by user
'''
resize = False
if user_res:
    resize = True
    resW, resH = int(user_res.split('x')[0]), int(user_res.split('x')[1])

'''
Video recording option (post inference) 
'''
if record:
    if source_type not in ['video','usb']:
        print('Recording only works for video and camera sources. Please try again.')
        sys.exit(0)
    if not user_res:
        print('Please specify resolution to record video at.')
        sys.exit(0)
        
    record_name = 'demo_empty.avi'
    record_fps = 30
    recorder = cv2.VideoWriter(record_name, cv2.VideoWriter_fourcc(*'MJPG'), record_fps, (resW,resH))

'''
Handles image, folder, video, or USB input
'''
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

'''
Colour scheme based on Tableu 10 Color Scheme
'''
bbox_colors = [(164,120,87), (68,148,228), (93,97,209), (178,182,133), (88,159,106), 
              (96,202,231), (159,124,168), (169,162,241), (98,118,150), (172,176,184)]

'''
Variables for fps tracking and frame handling
'''
avg_fps = 0
fps_buffer = []
fps_len = 100
img_count = 0
frame_idx = 0
csv_data = []

'''
Track class history
'''
track_class_history = defaultdict(lambda: deque(maxlen=30))

'''
Start of process
'''
while True:
    t_start = time.perf_counter()

    '''
    Read frame based on source type
    '''
    if source_type in ['image', 'folder']:
        if img_count >= len(imgs_list): break
        frame = cv2.imread(imgs_list[img_count]); img_count += 1
    else:
        ret, frame = cap.read()
        if not ret:
            break

    frame_id_str = f"frame_{frame_idx}" # unique ID for each frame
    frame_idx += 1

    '''
    Resize and pad image
    '''
    # frame = letterbox(frame, (640, 640))
    if resize == True:
        frame = cv2.resize(frame,(resW,resH))

    '''
    Rack model to detect and classify racks
    '''
    rack_results = rack_model.track(frame, persist=True, tracker='bytetrack.yaml', verbose=False) # Rack detection model and bytetrack tracking
    rack_boxes = rack_results[0].boxes # Contains bounding box information
    rack_data = []

    for box in rack_boxes:
        # print(f'Box here : {box}')
        xyxy = box.xyxy.cpu().numpy().squeeze()
        ymin, ymax = int(xyxy[1]), int(xyxy[3])
        xmin, xmax = int(xyxy[0]), int(xyxy[2])
        rack_data.append((ymin, ymax, xmin, xmax))

    rack_data.sort(key=lambda x: x[0])

    '''
    Bounding box and labels for racks
    '''
    for i, (ymin, ymax, xmin, xmax) in enumerate(rack_data):
        label = f"Rack {i+1}"
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 255), 2)
        cv2.putText(frame, label, (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)

    '''
    Object model to detect and classify objects
    '''
    obj_results = obj_model.track(frame, persist=True,tracker="bytetrack.yaml", verbose=False)
    obj_boxes = obj_results[0].boxes
    obj_data = []

    '''
    Process bounding box and track ID
    '''
    for box in obj_boxes:
        xyxy = box.xyxy.cpu().numpy().squeeze()
        xmin, ymin, xmax, ymax = map(int, xyxy)
        cx, cy = (xmin + xmax) // 2, (ymin + ymax) // 2
        original_cls_id = int(box.cls.item())
        conf = box.conf.item()
        if conf < min_thresh:
            continue
        track_id = int(box.id.item()) if box.id is not None else -1

        '''
        Identify most common class
        '''
        if track_id != -1:
            track_class_history[track_id].append(original_cls_id)
            cls_id = Counter(track_class_history[track_id]).most_common(1)[0][0]
        else:
            cls_id = original_cls_id

        obj_data.append((cx, cy, xmin, ymin, xmax, ymax, cls_id, conf, track_id))

    '''
    Sort object by cx - central x coordinate
    '''
    racks_objs = [[] for _ in range(len(rack_data))]

    for obj in obj_data:
        cx, cy, xmin, ymin, xmax, ymax, cls_id, conf, track_id = obj
        for rack_idx, (rack_ymin, rack_ymax, _, _) in enumerate(rack_data):
            if rack_ymin <= cy <= rack_ymax:
                racks_objs[rack_idx].append(obj)
                break

    '''
    Sort by cx but within rack
    '''
    for rack_objs in racks_objs:
        rack_objs.sort(key=lambda x: x[0], reverse=True)

    '''
    Debug information and save to CSV
    '''
    obj_count = 0
    for rack_idx, rack_objs in enumerate(racks_objs):
        #!! print(f"\nRack {rack_idx + 1}:")
        for i, (cx, cy, xmin, ymin, xmax, ymax, cls_id, conf, track_id) in enumerate(rack_objs):
            #!! print(f"  [{i}] cx: {cx}, cy: {cy}, xmin: {xmin}, ymin: {ymin}, xmax: {xmax}, ymax: {ymax}, cls_id: {cls_id}, name: {obj_labels[cls_id]} , conf: {conf:.3f} , track_id: {track_id}")

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

            '''
            Append detected object to CSV
            '''
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

    '''
    FPS Calculation
    '''
    fps = 1.0 / (time.perf_counter() - t_start)
    fps_buffer.append(fps)
    if len(fps_buffer) > fps_len:
        fps_buffer.pop(0)
    avg_fps = np.mean(fps_buffer)

    '''
    Display object count and fps
    '''
    cv2.putText(frame, f"FPS: {avg_fps:.2f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    cv2.putText(frame, f"Objects: {obj_count}", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    
    '''
    Show frame
    '''
    cv2.imshow("Rack & Object Detection", frame)
    if record: 
        recorder.write(frame)
        print("Recording frame...")

    '''
    Controls for frame navigation
    '''
    key = cv2.waitKey(0 if source_type in ['image', 'folder'] else 5)
    if key == ord('q'): break  # Exit on 'q'
    elif key == ord('s'): 
        cv2.waitKey()  # Save frame on 's'
    elif key == ord('p'): 
        cv2.imwrite('capture.png', frame)  # Save current frame as capture.png

'''
Release video usage
'''
if source_type in ['video', 'usb']: 
    cap.release()
if record: 
    recorder.release()

cv2.destroyAllWindows()

'''
Write CSV file for detection results
'''
with open(csv_file, mode='w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=csv_data[0].keys())
    writer.writeheader()
    writer.writerows(csv_data)

#!! Debug func
print("\nMost Common Classes by Track ID:")
for track_id in sorted(track_class_history.keys()):
    history = track_class_history[track_id]
    most_common = Counter(history).most_common(1)[0]
    class_id = most_common[0]
    class_name = obj_labels.get(class_id, f"Unknown_{class_id}")
    print(f"  Track {track_id}: {class_name} (Class ID: {class_id}) | Count: {most_common[1]}/{len(history)}")

print(f"Results saved to: {csv_file}")
print(f"Average FPS: {avg_fps:.2f}")

