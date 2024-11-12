import cv2
from ultralytics import YOLO
import numpy as np
import random

# Load YOLOv8 model for object detection (pre-trained)
model = YOLO("yolov8s.pt")  # You can choose other sizes like 'yolov8n.pt', 'yolov8l.pt', etc.

# Initialize the video capture object
video_capture = cv2.VideoCapture(0)

# Define possible sentiments for face detection
sentiments = ["Positive", "Neutral", "Negative"]

while True:
  # Capture frame-by-frame from the video feed
  ret, frame = video_capture.read()
  if not ret:
    break

  # Perform object detection using YOLOv8
  results = model(frame)  # Perform inference on the frame

  # Loop through detections
  for result in results:  # Loop through the list of Results objects
    # Extract detections for the current frame
    boxes = result.boxes.xyxy  # Get bounding boxes (xyxy format)
    confidences = result.boxes.conf  # Get confidence scores
    class_ids = result.boxes.cls  # Get class IDs

    # Loop through each detection
    for box, conf, class_id in zip(boxes, confidences, class_ids):
      x1, y1, x2, y2 = map(int, box)  # Get coordinates
      conf = float(conf)  # Confidence score
      class_id = int(class_id)  # Class ID

      # Only consider detections with confidence > 0.5
      if conf > 0.5:
        # Get the class name (e.g., 'person', 'cell phone', etc.)
        class_name = model.names[class_id]

        if class_name == 'person':  # If it's a person (face detected)
          # Draw bounding box for face
          cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

          # Assign a random sentiment for the face
          sentiment = random.choice(sentiments)
          sentiment_color = (0, 255, 0) if sentiment == "Positive" else (0, 255, 255) if sentiment == "Neutral" else (0, 0, 255)
          cv2.putText(frame, sentiment, (x2 - 10, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, sentiment_color, 1, cv2.LINE_AA)

        elif class_name == 'cell phone':  # If it's a cell phone
          # Draw bounding box for cell phone
          cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
          cv2.putText(frame, "Cell Phone", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

  # Display the frame with detected objects and sentiment
  cv2.imshow("Face & Cell Phone Detection with YOLOv8", frame)

  # Exit on pressing 'q'
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

# Release capture and close all windows
video_capture.release()
cv2.destroyAllWindows()
