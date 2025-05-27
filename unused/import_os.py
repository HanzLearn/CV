import os
import re

# Set the path to your folder
folder_path = r"C:\Users\ilhan\Desktop\Sem 8\FYP MOBILE ROBOT\yolo\mydin"  # folder path
prefix = "db_"
image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff")
video_extensions = (".avi", ".mp4", ".mov", ".mkv", ".wmv")

# Get list of all files in the folder
files = os.listdir(folder_path)

# Collect all existing db_N filenames
used_numbers = []
pattern = re.compile(rf"{prefix}(\d+)\.(\w+)$")

for f in files:
    match = pattern.match(f)
    if match:
        used_numbers.append(int(match.group(1)))

# Start numbering from the next available number
next_number = max(used_numbers, default=0) + 1

# Rename non-db_ files with an image extension
for f in files:
    if f.lower().endswith(image_extensions) and not pattern.match(f):
        ext = os.path.splitext(f)[1]  # Get file extension, e.g., .jpg
        new_name = f"{prefix}{next_number}{ext}"
        src = os.path.join(folder_path, f)
        dst = os.path.join(folder_path, new_name)
        os.rename(src, dst)
        print(f"Renamed: {f} -> {new_name}")
        next_number += 1
