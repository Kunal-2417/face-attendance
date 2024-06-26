from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
import cv2
import pickle
import numpy as np
import os
import csv
import time
from datetime import datetime
from win32com.client import Dispatch

def speak(text):
    speak = Dispatch("SAPI.SpVoice")
    speak.Speak(text)

video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier("data/haarcascade_frontalface_default.xml")

with open("data/names.pkl", "rb") as f:
    LABELS = pickle.load(f)
with open("data/faces_data.pkl", "rb") as f:
    FACES = pickle.load(f)

# Convert labels to numeric representations
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(LABELS)

# Convert FACES data to 2D array
FACES = np.array(FACES).reshape(len(FACES), -1)

knn = KNeighborsClassifier(n_neighbors=5)
# Use the encoded labels for fitting the model
knn.fit(FACES, encoded_labels)

imgbackground = cv2.imread("./Untitled.png")

COL_NAME = ['NAME', 'TIME']

while True:
    ret, frame = video.read()
    col = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(
        col,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    for (x, y, w, h) in faces:
        crop_img = frame[y:y + h, x:x + w]
        resized_img = cv2.resize(crop_img, (50, 50))  # Resize to match the classifier's expectations
        output = knn.predict(resized_img.flatten().reshape(1, -1))  # Flatten and reshape to 1D array
        predicted_label = label_encoder.inverse_transform(output)
        ts = time.time()
        date = datetime.fromtimestamp(ts).strftime("%d-%m-%y")
        timestamp = datetime.fromtimestamp(ts).strftime("%H-%M-%S")
        exist = os.path.isfile("Attendance/Attendance_" + date + ".csv")
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 2)
        cv2.rectangle(frame, (x, y - 40), (x + w, y), (50, 50, 255), -1)
        cv2.putText(frame, str(predicted_label[0]), (x, y - 15), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0))
        attendance = [str(predicted_label[0]), str(timestamp)]

    new_width = 420  # Set your desired width
    new_height = 320  # Set your desired height

    frame_resized = cv2.resize(frame, (new_width, new_height))

    # Adjust the X and Y positions to reposition the video feed
    x_position = 560  # Example X-axis shift
    y_position = 205  # Example Y-axis shift

    imgbackground[y_position:y_position + new_height, x_position:x_position + new_width] = frame_resized

    cv2.imshow("Frame", imgbackground)

    k = cv2.waitKey(1)
    if k == ord('p') or k == ord('P'):
        speak("Attendance Taken..")
        time.sleep(5)
        if exist:
            with open("Attendance/Attendance_" + date + ".csv", "+a") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(attendance)
        else:
            with open("Attendance/Attendance_" + date + ".csv", "+a") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(COL_NAME)
                writer.writerow(attendance)
    if k == ord("a"):
        break

video.release()
cv2.destroyAllWindows()
