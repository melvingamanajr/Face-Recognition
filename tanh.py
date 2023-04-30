import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import pickle
import mysql.connector
from mysql.connector import Error

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
    
    
cam_location = 'Library'
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
print(len(classNames))

def tanh(x):
    return np.tanh(x)   

def findEncodings(images):
    encodeList = []
    if os.path.exists('encodings_cache.pkl'):
        with open('encodings_cache.pkl', 'rb') as f:
            encodeList = pickle.load(f)
    else:
        for img in images:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            encode = face_recognition.face_encodings(img)[0]
            encodeList.append(tanh(encode))
            print(encodeList)
        with open('encodings_cache.pkl', 'wb') as f:
            pickle.dump(encodeList, f)
    return encodeList

def markLocation(name):
     with open('saved_files/markedlocation.csv','r+') as f:
         myDataList = f.readlines()
         nameList = []
         for line in myDataList:
             entry = line.split(',')
             nameList.append(entry[0])
         if name not in nameList:
             f.writelines(f'\n{name},{cam_location},{dtFormat},{dtString}')
              
encodeListKnown = findEncodings(images)
print('Encoding Complete')

def saveToDatabase(name, img_encoded, cam_location, dtFormat):
    sql = "INSERT INTO tblstudent (name, image, location, datetime) VALUES (%s, %s, %s, %s)"
    val = (name, img_encoded, cam_location, dtFormat)
    cursor.execute(sql, val)
    mydb.commit()  

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    
    imgS = cv2.resize(img,(0,0),None,0.25,0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    
    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS,facesCurFrame)
    
    for encodeFace,faceLoc in zip(encodesCurFrame,facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
        facesDis = face_recognition.face_distance(encodeListKnown,encodeFace)
        matchIndex = np.argmin(facesDis)
        
        if facesDis[matchIndex] < 0.4:
            name = classNames[matchIndex].upper()
            percentage = (1 - facesDis[matchIndex]) * 100                                                                                           + 20
            identified = f"{name} ({percentage:.2f}%)"
            print(f"Match found: {name} ({percentage:.2f}%)")
            top, right, bottom, left = faceLoc
            top, right, bottom, left = top * 4, right * 4, bottom * 4, left * 4
            cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(img, identified, (left + 6, top - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            markLocation(name)
            
            img_encoded = mysql.connector.Binary(encodeFace.tobytes())
            
            sql = "SELECT * FROM tblstudent WHERE name = %s AND datetime = %s"
            val = (name, dtFormat)
            cursor.execute(sql, val)
            result = cursor.fetchone()
            if not result:
                saveToDatabase(name, img_encoded, cam_location, dtFormat)
                print('Execute success!')
        else:
            top, right, bottom, left = faceLoc
            top, right, bottom, left = top * 4, right * 4, bottom * 4, left * 4
            cv2.rectangle(img, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.putText(img,"Unknown",(left + 6, top - 6),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0, 0, 255),2)

    cv2.imshow('Image',img)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()