from ultralytics import YOLO
import cv2
import mediapipe as mp

model = YOLO(r"C:\ASL recognition\runs\detect\runs\train\asl_custom\weights\best.pt")

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.5,
)

PAD = 0.5

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    
    display = frame.copy()          
    h, w = frame.shape[:2]

    result = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if result.multi_hand_landmarks:
        lm = result.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(display, lm, mp_hands.HAND_CONNECTIONS)

        xs = [p.x * w for p in lm.landmark]
        ys = [p.y * h for p in lm.landmark]
        cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
        side = max(max(xs) - min(xs), max(ys) - min(ys)) * (1 + 2 * PAD)

        x1, x2 = int(max(cx - side / 2, 0)), int(min(cx + side / 2, w))
        y1, y2 = int(max(cy - side / 2, 0)), int(min(cy + side / 2, h))
        cv2.rectangle(display, (x1, y1), (x2, y2), (255, 200, 0), 2)

        crop = frame[y1:y2, x1:x2]
        if crop.size > 0:
            r = model(crop, conf=0.3, verbose=False)[0]
            if len(r.boxes) > 0:
                best = r.boxes[r.boxes.conf.argmax()]
                label = f"{r.names[int(best.cls)]} {float(best.conf):.2f}"
                cv2.putText(display, label, (x1, max(y1 - 10, 25)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("ASL Webcam Test", display)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()