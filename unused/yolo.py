from ultralytics import YOLO

model = YOLO('rack.pt')
results = model('test.MOV', show=True, save=True)