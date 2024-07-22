import cv2
import face_recognition
import os
import numpy as np

def load_images_from_folder(path):
    images = []
    classNames = []

    # Loop through the directories
    for person_dir in os.listdir(path):
        person_path = os.path.join(path, person_dir)
        if os.path.isdir(person_path):
            # Load all images for this person
            person_images = []
            for image_file in os.listdir(person_path):
                image_path = os.path.join(person_path, image_file)
                curImg = cv2.imread(image_path)
                person_images.append(curImg)
            # Add the person's images and name to the dataset
            images.extend(person_images)
            classNames.extend([person_dir] * len(person_images))
    return images, classNames

def encode_faces(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encoded_faces = face_recognition.face_encodings(img)
        if len(encoded_faces) > 0:
            encoded_face = encoded_faces[0]
            encodeList.append(encoded_face)
    return encodeList
