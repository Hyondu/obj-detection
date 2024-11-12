import cv2
import random

# Load the pre-trained Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize the video capture object (0 for default camera)
video_capture = cv2.VideoCapture(0)

# Define possible sentiments
sentiments = ["Positive", "Neutral", "Negative"]

while True:
  # Capture frame-by-frame from the video feed
  ret, frame = video_capture.read()

  # Convert the frame to grayscale for face detection
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

  # Detect faces in the frame
  faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

  # Draw rectangles around the detected faces and display sentiment metric
  for (x, y, w, h) in faces:
    # Draw rectangle around face
    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Randomly assign a sentiment for demonstration
    sentiment = random.choice(sentiments)

    # Define position for the sentiment text (top-right corner of the rectangle)
    text_position = (x + w - 10, y - 10)

    # Set font and color for the sentiment label
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    color = (0, 255, 0) if sentiment == "Positive" else (0, 255, 255) if sentiment == "Neutral" else (0, 0, 255)

    # Put sentiment text on the frame
    cv2.putText(frame, sentiment, text_position, font, font_scale, color, 1, cv2.LINE_AA)

  # Display the frame with detected faces and sentiment
  cv2.imshow('Real-Time Face Tracking with Sentiment', frame)

  # Break the loop if 'q' is pressed
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

# Release the capture and close all OpenCV windows
video_capture.release()
cv2.destroyAllWindows()
