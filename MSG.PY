import cv2
import dlib
import numpy as np
from scipy.spatial import distance
import smtplib
from email.mime.text import MIMEText
import time
import os
import pygame

# Initialize pygame for sound
pygame.mixer.init()
alert_sound = "alarm.wav"  # Path to alert sound file
pygame.mixer.music.load(alert_sound)

# Constants for EAR and MAR thresholds
EAR_THRESHOLD = 0.25
MAR_THRESHOLD = 0.5
ALERT_COUNT_THRESHOLD = 3

# Email function to send SMS via Gmail
def send_sms_via_email(phone_number, carrier_gateway, message):
    # Email credentials (use environment variables for security)
    sender_email = os.getenv("SENDER_EMAIL", "your_email@gmail.com")
    sender_password = os.getenv("SENDER_PASSWORD", "your_app_password")
    
    # Create the email content
    recipient = f"{phone_number}@{carrier_gateway}"
    msg = MIMEText(message)
    msg['From'] = sender_email
    msg['To'] = recipient
    msg['Subject'] = "Drowsiness Alert"

    try:
        # Connect to Gmail's SMTP server
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()  # Secure the connection
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, recipient, msg.as_string())
            print("SMS sent successfully!")
    except Exception as e:
        print(f"Failed to send SMS: {e}")

# Calculate Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# Calculate Mouth Aspect Ratio (MAR)
def mouth_aspect_ratio(mouth):
    A = distance.euclidean(mouth[2], mouth[10])
    B = distance.euclidean(mouth[4], mouth[8])
    C = distance.euclidean(mouth[0], mouth[6])
    return (A + B) / (2.0 * C)

# Initialize Dlib's face detector and facial landmarks predictor
detector = dlib.get_frontal_face_detector()

predictor = dlib.shape_predictor(r"C:\\Users\\Dell\\Downloads\\vigileye\\vigil\\Model\\shape_predictor_68_face_landmarks (1).dat")

# Initialize webcam
cap = cv2.VideoCapture(0)
alert_count = 0
last_alert_time = time.time()

# Recipient's phone number and carrier gateway
phone_number = "9343241003"  # Replace with the recipient's phone number
carrier_gateway = "vtext.com"  # Adjust this based on the carrier

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    
    for face in faces:
        landmarks = predictor(gray, face)
        
        # Extract eye and mouth coordinates
        left_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)])
        right_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)])
        mouth = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(48, 68)])
        
        # Compute EAR and MAR
        ear_left = eye_aspect_ratio(left_eye)
        ear_right = eye_aspect_ratio(right_eye)
        ear = (ear_left + ear_right) / 2.0
        mar = mouth_aspect_ratio(mouth)
        
        # Highlight mouth landmarks
        for point in mouth:
            cv2.circle(frame, tuple(point), 2, (0, 255, 0), -1)
        
        # Check for drowsiness
        if ear < EAR_THRESHOLD or mar > MAR_THRESHOLD:
            alert_count += 1
            cv2.putText(frame, f"Drowsy! Alert count: {alert_count}", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Play alert sound and send SMS after multiple alerts
            if alert_count >= ALERT_COUNT_THRESHOLD and time.time() - last_alert_time > 60:
                pygame.mixer.music.play()
                send_sms_via_email(phone_number, carrier_gateway, "Drowsiness detected! Please wake up.")
                last_alert_time = time.time()
                alert_count = 0  # Reset alert count
        else:
            alert_count = 0  # Reset alert count if not drowsy
        
    # Display the frame with annotations
    cv2.imshow("Drowsiness Detection", frame)
    
    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()
