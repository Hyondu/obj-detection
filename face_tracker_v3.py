import cv2
import numpy as np
import random

# Load YOLO model
yolo_net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = yolo_net.getLayerNames()
output_layers = [layer_names[i - 1] for i in yolo_net.getUnconnectedOutLayers()]

# Load COCO class labels
with open("coco.names", "r") as f:
  classes = f.read().strip().split("\n")

# Initialize the video capture object (0 for default camera)
video_capture = cv2.VideoCapture(0)

# Define possible sentiments
sentiments = ["Positive", "Neutral", "Negative"]

while True:
  # Capture frame-by-frame from the video feed
  ret, frame = video_capture.read()
  
  height, width = frame.shape[:2]

  # Prepare the frame for YOLO model
  blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
  yolo_net.setInput(blob)
  detections = yolo_net.forward(output_layers)

  # Loop through YOLO detections
  for detection in detections:
    for obj in detection:
      scores = obj[5:]
      class_id = np.argmax(scores)
      confidence = scores[class_id]

      # Detect faces and cell phones with high confidence
      if confidence > 0.5 and (classes[class_id] == "person" or classes[class_id] == "cell phone"):
        # Calculate bounding box
        box = obj[0:4] * np.array([width, height, width, height])
        center_x, center_y, w, h = box.astype("int")
        x = int(center_x - w / 2)
        y = int(center_y - h / 2)

        # Determine label and color
        if classes[class_id] == "person":
          label = "Face"
          color = (255, 0, 0)  # Blue for face
          # Assign a random sentiment
          sentiment = random.choice(sentiments)
          sentiment_color = (0, 255, 0) if sentiment == "Positive" else (0, 255, 255) if sentiment == "Neutral" else (0, 0, 255)
          cv2.putText(frame, sentiment, (x + w - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, sentiment_color, 1, cv2.LINE_AA)
        else:
          label = "Cell Phone"
          color = (0, 255, 0)  # Green for cell phone

        # Draw bounding box and label
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

  # Display the frame with detected objects and sentiment
  cv2.imshow('YOLO Face and Cell Phone Detection with Random Sentiment', frame)

  # Break the loop if 'q' is pressed
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

# Release the capture and close all OpenCV windows
video_capture.release()
cv2.destroyAllWindows()
