import cv2
import dlib
import numpy as np
from scipy.spatial import distance
import time
from clicksend.client import ClickSendClient
from clicksend.controllers.sms_controller import SmsController
from clicksend.models import SmsMessage
from clicksend.configuration import Configuration

# Constants for EAR and MAR thresholds
EAR_THRESHOLD = 0.25
MAR_THRESHOLD = 0.5
ALERT_COUNT_THRESHOLD = 3

# ClickSend configuration (replace with your credentials)
api_key = "your_clicksend_api_key"  # Replace with your actual ClickSend API key
sms_sender = "YourSender"  # Your ClickSend sender ID (or phone number)

# Send SMS function using ClickSend API
def send_sms_via_clicksend(phone_number, message):
    configuration = Configuration()
    configuration.username = api_key
    clicksend_client = ClickSendClient(configuration)
    sms_controller = SmsController(clicksend_client)
    
    # Prepare the message payload
    sms_message = SmsMessage(
        to=phone_number,
        body=message,
        source='python'
    )
    
    # Send the SMS
    try:
        sms_controller.send(sms_message)
        print("SMS sent successfully via ClickSend!")
    except Exception as e:
        print(f"Failed to send SMS: {e}")

# Calculate Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Calculate Mouth Aspect Ratio (MAR)
def mouth_aspect_ratio(mouth):
    A = distance.euclidean(mouth[2], mouth[10])
    B = distance.euclidean(mouth[4], mouth[8])
    C = distance.euclidean(mouth[0], mouth[6])
    mar = (A + B) / (2.0 * C)
    return mar

# Initialize Dlib's face detector and facial landmarks predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("C:\\Users\\Dell\\Downloads\\vigileye\\vigil\\Model\\shape_predictor_68_face_landmarks (1).dat")

# Initialize webcam
cap = cv2.VideoCapture(0)
alert_count = 0
last_alert_time = time.time()

# Your recipient's phone number (make sure to include country code, e.g., +1 for USA)
phone_number = "1234567890"  # Replace with the recipient's phone number (including country code)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = detector(gray)
    
    for face in faces:
        landmarks = predictor(gray, face)
        
        # Get the coordinates of the eyes and mouth
        left_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)])
        right_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)])
        mouth = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(48, 68)])
        
        # Compute EAR and MAR
        ear_left = eye_aspect_ratio(left_eye)
        ear_right = eye_aspect_ratio(right_eye)
        mar = mouth_aspect_ratio(mouth)
        
        # Average EAR of both eyes
        ear = (ear_left + ear_right) / 2.0
        
        # Check if drowsiness is detected
        if ear < EAR_THRESHOLD or mar > MAR_THRESHOLD:
            alert_count += 1
            cv2.putText(frame, "Drowsy! Alert count: {}".format(alert_count), (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Send SMS after 3 alerts
            if alert_count >= ALERT_COUNT_THRESHOLD and time.time() - last_alert_time > 60:
                send_sms_via_clicksend(phone_number, "Drowsiness detected! Please wake up.")
                last_alert_time = time.time()
                alert_count = 0  # Reset alert count after sending the message
        else:
            alert_count = 0  # Reset alert count if not drowsy
        
    # Show the frame with the results
    cv2.imshow("Drowsiness Detection", frame)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()