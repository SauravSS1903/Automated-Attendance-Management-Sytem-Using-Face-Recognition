import cv2
import face_recognition
import pickle
import numpy as np
from datetime import datetime

students = ['riswi', 'saurav', 'bala']

# Load precomputed encodings
with open('encodings.pkl', 'rb') as f:
    encoded_face_train, classNames = pickle.load(f)

def markAttendance(name):
    with open(r'C:\Users\USER\OneDrive\Desktop\CLG\Face recognition\Attendance.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = [entry.split(',')[0] for entry in myDataList]
        if name not in nameList:
            now = datetime.now()
            time = now.strftime('%I:%M:%S:%p')
            date = now.strftime('%d-%B-%Y')
            f.writelines(f'{name}, {time}, {date}\n')

def markAbsent(name):
    with open(r'C:\Users\USER\OneDrive\Desktop\CLG\Face recognition\Attendance.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = [entry.split(',')[0] for entry in myDataList]
        if name not in nameList:
            now = datetime.now()
            time = now.strftime('%I:%M:%S:%p')
            date = now.strftime('%d-%B-%Y')
            f.writelines(f'{name}, {time}, {date}, absent\n')

cap = cv2.VideoCapture(0)  # Starting the webcam
if not cap.isOpened():
    print("Error: Camera could not be opened.")
else:
    recorded_students = set()  # Use a set to store marked students
    while True:
        success, img = cap.read()
        if not success:
            print("Failed to grab frame")
            break  # Exit the loop if frames cannot be grabbed

        imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
        faces_in_frame = face_recognition.face_locations(imgS)
        encoded_faces = face_recognition.face_encodings(imgS, faces_in_frame)

        for encode_face, faceloc in zip(encoded_faces, faces_in_frame):
            matches = face_recognition.compare_faces(encoded_face_train, encode_face)
            faceDist = face_recognition.face_distance(encoded_face_train, encode_face)
            matchIndex = np.argmin(faceDist)
            name = "Unknown"
            color = (0, 0, 255)  # Red

            if matches[matchIndex]:
                match_confidence = 1 - faceDist[matchIndex]
                if match_confidence >= 0.55:  # Adjust the threshold as needed
                    name = classNames[matchIndex].upper().lower()
                    color = (0, 255, 0)  # Green

            y1, x2, y2, x1 = faceloc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4  # Scale back up to the original size
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), color, cv2.FILLED)

            if color == (0, 255, 0):  # Green (recognized face)
                cv2.putText(img, name, (x1 + 6, y2 - 5), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            else:  # Red (unknown face)
                cv2.putText(img, "Unknown", (x1 + 6, y2 - 5), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

            if color == (0, 255, 0) and name not in recorded_students:  # Check if student has already been marked
                recorded_students.add(name)  # Add student to the set of marked students
                markAttendance(name)

        cv2.imshow('Webcam', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break  # Exit the loop if 'q' is pressed
        
    # Release the VideoCapture object and close windows
    cap.release()
    cv2.destroyAllWindows()

    for student in students:
        if student not in recorded_students:
            markAbsent(student)
