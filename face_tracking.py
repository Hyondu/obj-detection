import cv2

# Load the pre-trained Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize the video capture object (0 for default camera)
video_capture = cv2.VideoCapture(0)

while True:
  # Capture frame-by-frame from the video feed
  ret, frame = video_capture.read()

  # Convert the frame to grayscale for face detection
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

  # Detect faces in the frame
  faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

  # Draw rectangles around the detected faces
  for (x, y, w, h) in faces:
    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

  # Display the frame with detected faces
  cv2.imshow('Real-Time Face Tracking', frame)

  # Break the loop if 'q' is pressed
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

# Release the capture and close all OpenCV windows
video_capture.release()
cv2.destroyAllWindows()
