import cv2
import dlib
import numpy as np
import os
import csv
from datetime import datetime
from scipy.spatial import distance as dist
from imutils import face_utils
import pygame
import matplotlib.pyplot as plt

pygame.mixer.init()

# Constants and Thresholds
EAR_THRESHOLD = 0.25
MAR_THRESHOLD = 0.7
CONSEC_FRAMES = 20
ALERT_THRESHOLD = 3  # Number of detections before sending an alert
DATASET_DIR = "drowsiness_data"
CSV_FILE = "drowsiness_log.csv"

# Functions
def assure_path_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def mouth_aspect_ratio(mouth):
    A = dist.euclidean(mouth[3], mouth[9])
    B = dist.euclidean(mouth[2], mouth[10])
    C = dist.euclidean(mouth[0], mouth[6])
    return (A + B) / (2.0 * C)

# Initialize dlib’s face detector and landmark predictor
print("[INFO] Loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(r"C:\\Users\\Dell\\Downloads\\vigileye\\vigil\\Model\\shape_predictor_68_face_landmarks (1).dat")

# Facial Landmark Indices
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

# Initialize the video stream from the laptop's default webcam
assure_path_exists(DATASET_DIR)
cap = cv2.VideoCapture(0)  # Use '0' for the default webcam

# Data logging
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "Drowsiness", "Yawning"])

frame_count = 0
drowsiness_count = 0
yawn_count = 0
consecutive_drowsy_alerts = 0

def send_alert():
    print("[ALERT] Sending alert message...")  # Replace with actual messaging integration
    pygame.mixer.music.load("alarm.wav")  # Replace with your file path
    pygame.mixer.music.play()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        shape = predictor(gray, face)
        shape = face_utils.shape_to_np(shape)

        # Extract eye and mouth landmarks
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        mouth = shape[mStart:mEnd]

        # Calculate EAR and MAR
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0
        mar = mouth_aspect_ratio(mouth)

        # Detect Drowsiness
        if ear < EAR_THRESHOLD:
            frame_count += 1
            if frame_count >= CONSEC_FRAMES:
                drowsiness_count += 1
                consecutive_drowsy_alerts += 1
                cv2.putText(frame, "DROWSINESS ALERT!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                pygame.mixer.music.load("alarm.wav")  # Replace with your file path
                pygame.mixer.music.play()
                
                # Save frame
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                cv2.imwrite(os.path.join(DATASET_DIR, f"drowsiness_{timestamp}.jpg"), frame)
                
                # Log data
                with open(CSV_FILE, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([datetime.now(), "Drowsiness", "No"])
                
                frame_count = 0  # Reset frame counter
                
                # Send alert after three consecutive drowsiness detections
                if consecutive_drowsy_alerts >= ALERT_THRESHOLD:
                    send_alert()
                    consecutive_drowsy_alerts = 0  # Reset alert counter
        else:
            frame_count = 0  # Reset if eyes open

        # Detect Yawning
        if mar > MAR_THRESHOLD:
            yawn_count += 1
            cv2.putText(frame, "YAWNING ALERT!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            pygame.mixer.music.load("alarm.wav")  # Replace with your file path
            pygame.mixer.music.play()
            
            # Save frame
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            cv2.imwrite(os.path.join(DATASET_DIR, f"yawn_{timestamp}.jpg"), frame)
            
            # Log data
            with open(CSV_FILE, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([datetime.now(), "No", "Yawning"])

        # Draw facial landmarks
        for (x, y) in np.concatenate((leftEye, rightEye, mouth)):
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

    cv2.imshow("Driver Drowsiness Detection", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
