"""
YOLO Object Detection - LIVE Webcam
80 cheezein detect karta hai: insaan, gari, kutta, kursi, mobile...
"""

import cv2
from ultralytics import YOLO

print("YOLO model load ho raha hai...")
model = YOLO("yolov8s.pt")   # small = zyada accurate! (pehli dafa download hoga)
print("Model ready!")

cap = cv2.VideoCapture(0)
print("Webcam ON! Cheezein dikhao - band karne ke liye 'q' dabao!")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO se detect karo
    results = model(frame, verbose=False)

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = f"{model.names[cls]} {conf:.0%}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("YOLO Detection ('q' = band)", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print("Ho gaya!")
