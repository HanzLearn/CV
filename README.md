# YOLO Based Planogram Compliance System

## YOLO Model For Item and Rack Detection

Trained YOLOv11n for item and rack detection. Limited number of rack to 3.

* Annotated using label-studio and trained in google colab.
* Utilized bytetrack as a tracking algorithm to keep track of items and easily manipulate bounding boxes(bbox).
* Bbox data is then used as a sorting parameter. Easily localize detected objects.
* OpenCV is used to manipulate sources and annotate the video.
* Detected items are then kept in a dataframe in order of different racks. e.g All items in Rack_1 are stored first then Rack_2

## Post Processing using Pandas

Data on bbox and item class are stored in a dataframe. Then data is kept in different dataframes depending on rack number.

* Makes post processing easier as data is kept in separate dataframes.
* Allows easier planogram generation as data is group in separate dataframes.
* Uses a threshold value to determine whether an item is valid. Detections over threshold value is stored while detections below are discarded.
* Blank rows are added between header and change of racks. e.g. an empty row between the last object of Rack_1 and first object of Rack_2

## Algorithm Development for Planogram Compliance

An algorithm is developed to determine how many products are missing from the expected planogram.

* Main Case: When the item is fully empty from the shelf. i.e no item of the same class is present
* Edge Case 1: When the item is the first item on the shelf and is not fully empty. i.e. there are items of the same class present
* Edge Case 2: When the item is the last item on the shelf and is not fully empty. i.e. there are items of the same class present

Based on the detected planogram, i.e. the planogram generated from the YOLO model, this algorithm will determine how many of the specific class items are missing and generate a concise report on how many should be placed on the shelf. It does not specify where the item on the shelf should be, just how many should be restocked.

## Report Generation in Excel

The algorithm used will then generate a report in Excel showing how many items are missing and will show the difference between the generated planogram and the expected planogram. It will show where the item is missing and how many items of each class is missing.
