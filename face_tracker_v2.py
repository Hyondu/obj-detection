import cv2
import numpy as np
import random

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Load YOLO model for cell phone detection
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

  # Get frame dimensions
  height, width = frame.shape[:2]

  # Convert frame to grayscale for face detection with Haar Cascade
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

  # Detect faces
  faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

  # Draw rectangles and assign random sentiments for detected faces
  for (x, y, w, h) in faces:
    # Draw face bounding box
    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Assign a random sentiment
    sentiment = random.choice(sentiments)

    # Display sentiment at the top-right corner of the face rectangle
    text_position = (x + w - 10, y - 10)
    font_scale = 1.0
    sentiment_color = (0, 255, 0) if sentiment == "Positive" else (0, 255, 255) if sentiment == "Neutral" else (0, 0, 255)
    cv2.putText(frame, sentiment, text_position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, sentiment_color, 1, cv2.LINE_AA)

  # Prepare frame for YOLO model for cell phone detection
  blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
  yolo_net.setInput(blob)
  detections = yolo_net.forward(output_layers)

  # Loop through YOLO detections
  for detection in detections:
    for obj in detection:
      scores = obj[5:]
      class_id = np.argmax(scores)
      confidence = scores[class_id]

      # Detect "cell phone" class with high confidence
      if confidence > 0.5 and classes[class_id] == "cell phone":
        # Calculate bounding box for cell phone
        box = obj[0:4] * np.array([width, height, width, height])
        center_x, center_y, w, h = box.astype("int")
        x = int(center_x - w / 2)
        y = int(center_y - h / 2)

        # Draw bounding box for cell phone
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, "Cell Phone", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

  # Display the frame with detected objects and sentiment
  cv2.imshow('Face Detection with Random Sentiment & Cell Phone Detection', frame)

  # Break the loop if 'q' is pressed
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

# Release the capture and close all OpenCV windows
video_capture.release()
cv2.destroyAllWindows()
