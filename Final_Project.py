import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import pickle
import mysql.connector
from mysql.connector import Error
from sklearn.svm import SVC

try:
    mydb = mysql.connector.connect(
      host="localhost",
      user="root",
      password="",
      database="dbtraces"
    )
    if mydb.is_connected():
        db_Info = mydb.get_server_info()
        print("Connected to MySQL database... MySQL Server version on ",db_Info)
        cursor = mydb.cursor()
        cursor.execute("select database();")
        record = cursor.fetchone()
        print("Your connected to database: ", record)

except Error as e:
    print("Error while connecting to MySQL", e)
    
cam1 = 'Office'
cam2 = 'Library'

path = "student_images"
now = datetime.now()
dtFormat = now.strftime('%Y-%m-%d')
dtString = now.strftime('%H:%M:%S')
images = []
classNames = []
myList = os.listdir(path)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])

def findEncodings(images):
    encodeList = []
    if os.path.exists('encodings_cache.pkl'):
        with open('encodings_cache.pkl', 'rb') as f:
            encodeList = pickle.load(f)
    else:
        for img in images:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            encode = face_recognition.face_encodings(img)[0]
            encodeList.append(encode)
            print('Cache Image.. Please wait!..')
        with open('encodings_cache.pkl', 'wb') as f:
            pickle.dump(encodeList, f)
    return encodeList


encodeListKnown = findEncodings(images)
train_names = classNames

clf = SVC(kernel='linear', probability=True)
clf.fit(encodeListKnown, train_names)
print('Encoding Complete')

def saveToDatabase(name, img_encoded, cam_location, dtFormat):
    sql = "INSERT INTO tblstudent (name, image, location, datetime) VALUES (%s, %s, %s, %s)"
    val = (name, img_encoded, cam_location, dtFormat)
    cursor.execute(sql, val)
    mydb.commit()  

cap1 = cv2.VideoCapture(0)
cap2 = cv2.VideoCapture(1)

while True:
        success1, img1 = cap1.read()
        success2, img2 = cap2.read()

        # Resize and convert to RGB format
        imgS1 = cv2.resize(img1, (0, 0), None, 0.25, 0.25)
        imgS1 = cv2.cvtColor(imgS1, cv2.COLOR_BGR2RGB)

        imgS2 = cv2.resize(img2, (0, 0), None, 0.25, 0.25)
        imgS2 = cv2.cvtColor(imgS2, cv2.COLOR_BGR2RGB)

        # Find face locations and encodings
        facesCurFrame1 = face_recognition.face_locations(imgS1)
        encodesCurFrame1 = face_recognition.face_encodings(imgS1, facesCurFrame1)

        facesCurFrame2 = face_recognition.face_locations(imgS2)
        encodesCurFrame2 = face_recognition.face_encodings(imgS2, facesCurFrame2)

        # Loop over each face in the frame and compare to known faces
        for encodeFace, faceLoc in zip(encodesCurFrame1, facesCurFrame1):
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
            facesDis = face_recognition.face_distance(encodeListKnown, encodeFace)
            matchIndex = np.argmin(facesDis)

            if facesDis[matchIndex] < 0.5:
                name = classNames[matchIndex].upper()
                percentage = (1 - facesDis[matchIndex]) * 100
                identified = f"{name} ({percentage:.2f}%)"
                print(f"Match found on camera 1: {name} ({percentage:.2f}%)")
                top, right, bottom, left = faceLoc
                top, right, bottom, left = top * 4, right * 4, bottom * 4, left * 4
                cv2.rectangle(img1, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(img1, identified, (left + 6, top - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

                img_encoded = mysql.connector.Binary(encodeFace.tobytes())

                dtFormat = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                sql = "SELECT * FROM tblstudent WHERE name = %s AND datetime = %s"
                val = (name, dtFormat)
                cursor.execute(sql, val)
                result = cursor.fetchone()
                if not result:
                    saveToDatabase(name, img_encoded, cam1, dtFormat)
                    print('Execute success on camera 1!')
            else:
                top, right, bottom, left = faceLoc
                top, right, bottom, left = top * 4, right * 4, bottom * 4, left * 4
                cv2.rectangle(img1, (left, top), (right, bottom), (0, 0, 255), 2)
                
                # Loop over each face in the frame and compare to known faces (camera 2)
                for encodeFace, faceLoc in zip(encodesCurFrame2, facesCurFrame2):
                    matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
                    facesDis = face_recognition.face_distance(encodeListKnown, encodeFace)
                    matchIndex = np.argmin(facesDis)
            
                    if facesDis[matchIndex] < 0.5:
                        name = classNames[matchIndex].upper()
                        percentage = (1 - facesDis[matchIndex]) * 100
                        identified = f"{name} ({percentage:.2f}%)"
                        print(f"Match found on camera 2: {name} ({percentage:.2f}%)")
                        top, right, bottom, left = faceLoc
                        top, right, bottom, left = top * 4, right * 4, bottom * 4, left * 4
                        cv2.rectangle(img2, (left, top), (right, bottom), (0, 255, 0), 2)
                        cv2.putText(img2, identified, (left + 6, top - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            
                        img_encoded = mysql.connector.Binary(encodeFace.tobytes())
            
                        dtFormat = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        sql = "SELECT * FROM tblstudent WHERE name = %s AND datetime = %s"
                        val = (name, dtFormat)
                        cursor.execute(sql, val)
                        result = cursor.fetchone()
                        if not result:
                            saveToDatabase(name, img_encoded, cam2, dtFormat)
                            print('Execute success on camera 2!')
                    else:
                        top, right, bottom, left = faceLoc
                        top, right, bottom, left = top * 4, right * 4, bottom * 4, left * 4
                        cv2.rectangle(img2, (left, top), (right, bottom), (0, 0, 255), 2)
            
                # Display the resulting frames
                cv2.imshow('Camera 1', img1)
                cv2.imshow('Camera 2', img2)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the capture and destroy windows
cap1.release()
cap2.release()
cv2.destroyAllWindows()