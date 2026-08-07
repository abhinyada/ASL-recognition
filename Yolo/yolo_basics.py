from ultralytics import YOLO
import numpy 

model =YOLO("yolov8n.pt", "v8")  # load a pretrained model (recommended for training)


detection= model.predict(source="F:\ Vs code\ rigby cat.jpg", conf=0.25, save=True)  # predict on an imag
print(detection)  # print results
print(detection[0].numpy())  # print boxes coordinates (pixels)