import cv2
import numpy as np
import face_recognition
import dlib
import os
import pickle

# Load Haar Cascade Classifier for face detection
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Load DLIB's HOG-based face detector
hog_face_detector = dlib.get_frontal_face_detector()

# Set the threshold for the minimum distance between face encodings to consider a match
distance_threshold = 0.6

path = "student_images"
images = []
classNames = []
myList = os.listdir(path)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])

# Load face encodings from cache or calculate them from images in the student_images folder
def load_encodings():
    encodeList = []
    if os.path.exists('encodings_cache.pkl'):
        with open('encodings_cache.pkl', 'rb') as f:
            encodeList = pickle.load(f)
    else:
        for img in images:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            encode = face_recognition.face_encodings(img)[0]
            encodeList.append(encode)
            print('Calculating encodings.. Please wait!..')
        with open('encodings_cache.pkl', 'wb') as f:
            pickle.dump(encodeList, f)
    return encodeList

encodeListKnown = load_encodings()
print('Encoding Complete')

# Initialize video capture from default camera
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the video stream
    success, img = cap.read()
    img = cv2.flip(img, 1)

    # Convert the image to grayscale for Haar Cascade Classifier face detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces using Haar Cascade Classifier and DLIB's HOG-based face detector
    faces_haar = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    faces_dlib = hog_face_detector(gray, 1)

    # Convert the image to RGB for face_recognition face detection
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Detect faces using face_recognition face detection
    faces_recognition = face_recognition.face_locations(img_rgb)

    # Create a list to store the encodings and names of all the detected faces
    encodings = []
    names = []

    # Loop through all the detected faces and encode them using all three algorithms
    for (x, y, w, h) in faces_haar:
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]
        face = cv2.resize(roi_color, (48, 48))
        encode = face_recognition.face_encodings(face, [(y, x + w, y + h, x)])[0]
        encodings.append(encode)
        names.append('Haar Cascade Classifier')
        

        for face_rect in faces_dlib:
            x = face_rect.left()
            y = face_rect.top()
            w = face_rect.right() - x
            h = face_rect.bottom() - y
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = img[y:y + h, x:x + w]
            face = cv2.resize(roi_color, (48, 48))
            encode = face_recognition.face_encodings(face, [(y, x + w, y + h, x)])[0]
            face_encodings = face_recognition.face_encodings(img_rgb, [face_rect])
            if len(face_encodings) > 0:
                encodings.append(face_encodings[0])
                names.append('DLIB HOG Face Detector')

            for face_loc in faces_recognition:
                top, right, bottom, left = face_loc
                face = img[top:bottom, left:right]
                encode = face_recognition.face_encodings(face)[0]
                encodings.append(encode)
                names.append('face_recognition')
                
            
                for encoding, name in zip(encodings, names):
                    matches = face_recognition.compare_faces(encodeListKnown, encoding, tolerance=distance_threshold)
                    face_distances = face_recognition.face_distance(encodeListKnown, encoding)
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:              
                        name = classNames[best_match_index]
                        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.putText(img, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
                    
    cv2.imshow('Face Recognition', img)
    if cv2.waitKey(1) == ord('q'): 
        break
    
cap.release()
cv2.destroyAllWindows()