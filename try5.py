import cv2
import dlib
import numpy as np
import os
import csv
from datetime import datetime
from scipy.spatial import distance as dist
from imutils import face_utils
import pygame
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Initialize Pygame mixer
pygame.mixer.init()

# Constants and Thresholds
EAR_THRESHOLD = 0.25
MAR_THRESHOLD = 0.7
CONSEC_FRAMES = 20
DATASET_DIR = "drowsiness_data"
CSV_FILE = "drowsiness_log.csv"

# Email Configuration
SENDER_EMAIL = "sharanchandan5415@gmail.com"  # Replace with your email
EMAIL_PASSWORD = "dnjr yzpg kmar bces"# Replace with your app password
RECEIVER_SMS = "7204030048@smsgateway.com"  # Replace with recipient's SMS gateway address

# Helper Functions
def send_sms_alert():
    """Send an SMS alert using email-to-SMS gateway."""
    if not all([SENDER_EMAIL, EMAIL_PASSWORD, RECEIVER_SMS]):
        print("[ERROR] Missing email credentials or receiver SMS address.")
        return

    subject = "Drowsiness Alert"
    body = "Drowsiness detected! Please take a break."

    msg = MIMEMultipart()
    msg["From"] = SENDER_EMAIL
    msg["To"] = RECEIVER_SMS
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(SENDER_EMAIL, EMAIL_PASSWORD)
        server.sendmail(SENDER_EMAIL, RECEIVER_SMS, msg.as_string())
        server.quit()
        print("[INFO] SMS alert sent via email-to-SMS.")
    except Exception as e:
        print(f"[ERROR] Could not send SMS: {e}")

def assure_path_exists(path):
    """Ensure the directory exists."""
    if not os.path.exists(path):
        os.makedirs(path)

def eye_aspect_ratio(eye):
    """Calculate Eye Aspect Ratio (EAR)."""
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def mouth_aspect_ratio(mouth):
    """Calculate Mouth Aspect Ratio (MAR)."""
    A = dist.euclidean(mouth[3], mouth[9])
    B = dist.euclidean(mouth[2], mouth[10])
    C = dist.euclidean(mouth[0], mouth[6])
    return (A + B) / (2.0 * C)

# Initialize dlib’s face detector and landmark predictor
print("[INFO] Loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("C:\\Users\\Dell\\Downloads\\vigileye\\vigil\\Model\\shape_predictor_68_face_landmarks (1).dat")  # Update with your file path

# Facial landmark indices
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

# Ensure output directories exist
assure_path_exists(DATASET_DIR)
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "Drowsiness", "Yawning"])

# Initialize webcam
cap = cv2.VideoCapture(0)

frame_count = 0
drowsiness_count = 0
yawn_count = 0

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

        # Detect drowsiness
        if ear < EAR_THRESHOLD:
            frame_count += 1
            if frame_count >= CONSEC_FRAMES:
                drowsiness_count += 1
                cv2.putText(frame, "DROWSINESS ALERT!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                pygame.mixer.music.load("alarm.wav")  # Replace with the correct path to your sound file
                pygame.mixer.music.play()
                send_sms_alert()

                # Save frame
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                cv2.imwrite(os.path.join(DATASET_DIR, f"drowsiness_{timestamp}.jpg"), frame)

                # Log data
                with open(CSV_FILE, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([datetime.now(), "Drowsiness", "No"])
        else:
            frame_count = 0

        # Detect yawning
        if mar > MAR_THRESHOLD:
            yawn_count += 1
            cv2.putText(frame, "YAWNING ALERT!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            pygame.mixer.music.load("alarm.wav")
            pygame.mixer.music.play()

            # Save frame
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            cv2.imwrite(os.path.join(DATASET_DIR, f"yawn_{timestamp}.jpg"), frame)

            # Log data
            with open(CSV_FILE, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([datetime.now(), "No", "Yawning"])

    cv2.imshow("Driver Drowsiness Detection", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
