import cv2
from ultralytics import YOLO
import numpy as np
import random
import sys
import os
import yaml

from face_recognition.face_detection.scrfd.detector import SCRFD
from face_recognition.face_tracking.tracker.byte_tracker import BYTETracker
from face_recognition.face_tracking.tracker.visualize import plot_tracking

from deepface import DeepFace
# Suppress unnecessary logs from YOLOv8 and OpenCV
#os.environ['PYTHONWARNINGS'] = "ignore"  # Ignore Python warnings
#sys.stdout = open(os.devnull, 'w')  # Redirect standard output to null
#sys.stderr = open(os.devnull, 'w')  # Redirect error output to null


def load_config(file_name):
    with open(file_name, "r") as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)


def compute_iou(box1, box2):
    """
    Computes the Intersection over Union (IoU) between two bounding boxes.

    Parameters:
    - box1: list or array of [xmin, ymin, xmax, ymax]
    - box2: list or array of [xmin, ymin, xmax, ymax]

    Returns:
    - iou: float, Intersection over Union between box1 and box2
    """
    
    # Determine the coordinates of the intersection rectangle
    x_left   = max(box1[0], box2[0])
    y_top    = max(box1[1], box2[1])
    x_right  = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    # Compute the area of intersection rectangle
    if x_right < x_left or y_bottom < y_top:
        # No overlap between the boxes
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # Compute the area of both bounding boxes
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # Compute the union area
    union_area = box1_area + box2_area - intersection_area

    # Compute the IoU
    iou = intersection_area / union_area

    return iou

def compute_iou_boxs(face_box,boxes2):
  result = 0
  hand_result = [-1,-1,-1,-1]
  
  for b2 in boxes2:
      iou = compute_iou(face_box, b2)
      if iou > result:
        result = iou
        hand_result = b2
  return result, hand_result

# Load YOLOv8 model for object detection (pre-trained)
model = YOLO("yolov8s.pt")  # You can choose other sizes like 'yolov8n.pt', 'yolov8l.pt', etc.

# Initialize the video capture object
video_capture = cv2.VideoCapture(0 )

# Define possible sentiments for face detection
sentiments = ["Positive", "Neutral", "Negative"]

file_name = "./face_recognition/face_tracking/config/config_tracking.yaml"
config_tracking = load_config(file_name)
detector = SCRFD(model_file="face_recognition/face_detection/scrfd/weights/scrfd_2.5g_bnkps.onnx")
face_tracker = BYTETracker(args=config_tracking, frame_rate=30)
frame_id = 0

while True:
  # Capture frame-by-frame from the video feed
  ret, frame = video_capture.read()
  if not ret:
    break

  # face detection
  count = 0
  
  outputs, img_info, bboxes, landmarks = detector.detect_tracking(image=frame)
  if outputs is not None:
    # Perform object detection using YOLOv8
    hand_pred = model(frame)  # Perform inference on the frame
    hand_boxes = []
    
    # Loop through detections
    for result in hand_pred:  # Loop through the list of Results objects
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

          if class_name == 'cell phone':  # If it's a cell phone
            # Draw bounding box for cell phone
            cv2.rectangle(img_info["raw_img"], (x1, y1), (x2, y2), (0, 255, 0), 2)
            hand_boxes.append([x1,y1,x2,y2])
      
    face_preds = face_tracker.update(
        outputs, [img_info["height"], img_info["width"]], (128, 128)
    )
    online_tlwhs = []
    online_ids = []
    online_scores = []
    iou = 0
    hand_result = [-1,-1,-1,-1]
    for t in face_preds:
        tlwh = t.tlwh
        tid = t.track_id
        
        vertical = tlwh[2] / tlwh[3] > config_tracking["aspect_ratio_thresh"]
        if tlwh[2] * tlwh[3] > config_tracking["min_box_area"] and not vertical:
            online_tlwhs.append(tlwh)
            online_ids.append(tid)
            online_scores.append(t.score)
            iou, hand_result = compute_iou_boxs(t.tlbr_face, hand_boxes)
    online_im = plot_tracking(
        img_info["raw_img"],
        online_tlwhs,
        online_ids,
        iou = iou,
        hand_result = hand_result
    )
  else:
      online_im = img_info["raw_img"]
      

  # Display the frame with detected objects and sentiment
  cv2.imshow("Face & Cell Phone Detection with YOLOv8", online_im)
  frame_id += 1

  # Exit on pressing 'q'
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

# Release capture and close all windows
video_capture.release()
cv2.destroyAllWindows()