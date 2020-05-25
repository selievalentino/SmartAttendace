import os
import numpy as np
from PIL import Image
import cv2
import pickle

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Returns the directory where Face_train.py is located
image_dir = os.path.join(BASE_DIR, "images")
recognizer = cv2.face.LBPHFaceRecognizer_create()

current_id = 0
label_ids = {}
y_labels = []
x_train = []

for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.endswith("png") or file.endswith("jpg"):
            path = os.path.join(root, file)
            label = os.path.basename(os.path.dirname(path)).replace(" ", "-").lower()
            #print(label, path)
            # x_train.append(path)
            # y_labels.append(label)
            if not label in label_ids:
                label_ids[label] = current_id
                current_id += 1
            id_ = label_ids[label]
            pil_image = Image.open(path).convert("L")  # change to grayscale
            #size = (800, 800)
            #resizedImg = pil_image.resize(size, Image.ANTIALIAS)
            #image_array = np.array(resizedImg, "uint8")
            image_array = np.array(pil_image, "uint8")
            faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.5, minNeighbors=5)

            for (x, y, w, h) in faces:
                roi = image_array[x:x + w, y:y + h]
                x_train.append(roi)
                y_labels.append(id_)
with open("labels.pickle",'wb') as f:
    pickle.dump(label_ids, f)
recognizer.train(x_train, np.array(y_labels))
recognizer.save("trainer.yml")