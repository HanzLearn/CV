PYTHON=python 
SCRIPT=yolo_detect.py

MODEL=obj.pt 
SOURCE=test.mov

run:
	$(PYTHON) $(SCRIPT) --model $(MODEL) --source $(SOURCE)