from ultralytics import YOLO
import cv2

model = YOLO(r"F:\Vs code\runs\detect\runs\train\asl_custom\weights\best.pt")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    results = model(frame, conf=0.3, verbose=False)
    annotated_frame = results[0].plot()

    cv2.imshow("ASL Webcam Test", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()