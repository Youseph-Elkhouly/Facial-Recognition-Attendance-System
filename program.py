import face_recognition
import cv2
import numpy as np
import os
import csv
import datetime 

video_capture = cv2.VideoCapture(0)

# Load sample pictures and learn how to recognize them.
jobs_image = face_recognition.load_image_file("photos/HOWARD.jpg")
jobs_face_encoding = face_recognition.face_encodings(jobs_image)[0]

gates_image = face_recognition.load_image_file("photos/MIKE.jpg")
gates_face_encoding = face_recognition.face_encodings(gates_image)[0]

random_image = face_recognition.load_image_file("photos/raj.jpg")
random_face_encoding = face_recognition.face_encodings(random_image)[0]

# Create arrays of known face encodings and their names
known_face_encodings = [
    jobs_face_encoding,
    gates_face_encoding,
    random_face_encoding
]

known_face_names = [
    "HOWARD",
    "MIKE",
    "RAJ"
]

students = known_face_names.copy()

face_locations = []
face_encodings = []
face_names = []
s = True

now = datetime.datetime.now()
current_time = now.strftime("%Y-%m-%d")

f = open('attendance.csv', 'w+', newline='')
lnwriter = csv.writer(f)

while True:
    _, frame = video_capture.read()
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]
    
    if s:
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        face_names = []
        
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            
            name = "Unknown"
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
            
            face_names.append(name)
            if name in students:
                students.remove(name)
                print(students)
                current_time = datetime.datetime.now().strftime("%H-%M-%S")
                lnwriter.writerow([name, current_time])
    
    cv2.imshow('Attendance System', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
f.close()
