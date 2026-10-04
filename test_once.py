"""Ek tasveer lo, YOLO kya dekhta hai print karo!"""
import cv2
from ultralytics import YOLO

model = YOLO("yolov8s.pt")
print("Model load ho gaya!")

cap = cv2.VideoCapture(0)
ret, frame = cap.read()
cap.release()

# Bohat halka threshold - kamzor detection bhi pakro!
results = model(frame, verbose=False, conf=0.1)

print("--- YOLO ne ye dekha ---")
for r in results:
    for box in r.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        print(f"{model.names[cls]}: {conf:.0%}")
print("--- khatam ---")

cv2.imwrite("test_frame.jpg", frame)
print("test_frame.jpg save ho gayi!")
