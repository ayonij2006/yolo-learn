from ultralytics import YOLO
import cv2

# Load YOLOv8 model (use your custom model if needed)
model = YOLO("yolo11n.pt")  # or yolov8s.pt, yolov8m.pt, yolov8l.pt, etc.

# Open webcam
cap = cv2.VideoCapture(0)  # 0 = default camera

# Check if webcam is opened
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Run YOLO prediction on the frame
    results = model.predict(frame, imgsz=640, conf=0.5)

    # Annotate the frame with YOLO results
    annotated_frame = results[0].plot()

    # Show the output
    cv2.imshow("YOLOv11 Webcam", annotated_frame)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
