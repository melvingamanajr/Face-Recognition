import os
import cv2
import face_recognition

# Set the directory path containing the images
directory_path = "student_images"

# Loop through all the image files in the directory path
for file in os.listdir(directory_path):
    # Check if the file is an image
    if file.endswith('.jpg') or file.endswith('.png'):
        # Load the image file
        image_path = os.path.join(directory_path, file)
        image = cv2.imread(image_path)

        # Find the face locations in the image
        face_locations = face_recognition.face_locations(image, model='hog')

        # Iterate through each face and resize the image
        for top, right, bottom, left in face_locations:
            face_width = right - left
            face_height = bottom - top
            face_boundary = int(min(face_width, face_height) * 0.3)  # increase the face boundary

            top = max(top - face_boundary, 0)
            right = min(right + face_boundary, image.shape[1])
            bottom = min(bottom + face_boundary, image.shape[0])
            left = max(left - face_boundary, 0)

            face_image = image[top:bottom, left:right]
            resized_image = cv2.resize(face_image, (224, 224))

            # Save the resized image with the same filename in the same directory
            cv2.imwrite(image_path, resized_image)