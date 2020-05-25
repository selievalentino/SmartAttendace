import numpy as np
import cv2
import pickle
import requests
import os
import face_recognition


url = "http://10.249.136.123:8080/shot.jpg"
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")
labels = {}
with open("labels.pickle", 'rb') as f:
    org_labels = pickle.load(f)
    labels = {value: key for key, value in org_labels.items()}
img_counter = 0
user = "user_{}".format(img_counter)
while True:
    # imgResp = requests.get(url)
    # imgNp = np.array(bytearray(imgResp.content), dtype=np.uint8)
    # frame = cv2.imdecode(imgNp, -1)

    ret, frame = cap.read()  # ret returns true if read is success
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)

    for (x, y, w, h) in faces:
        # print(faces)
        # print("************************")
        roi = gray[y:y + h, x:x + w]

        id_, conf = recognizer.predict(roi)
        if conf >= 40 and conf<=100:
            print(labels[id_])
            print(conf)
            font = cv2.FONT_HERSHEY_COMPLEX_SMALL
            name = labels[id_]
            color = (255, 100, 0)
            stroke = 1
            cv2.putText(frame, name, (x, y), font, 1, color, stroke, cv2.LINE_AA)


        # cv2.imwrite(img_item,roi)

        color = (0, 255, 0)
        stroke = 1
        end_coordinate_x = x + w
        end_coordinate_y = y + h
        cv2.rectangle(frame, (x, y), (end_coordinate_x, end_coordinate_y), color, stroke)
    # dimension = (500, 500)
    # img = cv2.resize(frame, dimension, interpolation=cv2.INTER_AREA)
    cv2.imshow('Frame', frame)

    if (cv2.waitKey(20) & 0xff) == ord('c'):
        found = False
        file_name = ""
        path = os.getcwd()+"\\"+user
        try:

            for root, dirs, files in os.walk(os.getcwd()):
                if files == path:
                    found = True
                    os.chdir(os.getcwd())
            if not found:
                os.mkdir(path)


        except OSError:
            print("Creation of the directory %s failed" % path)
        else:
                print("Successfully created the directory %s " % path)

                img_name = "facedetect_webcam_{}.png".format(img_counter)
                cv2.imwrite(os.path.join(path, img_name), frame)
                print("{} written!".format(img_name))
                img_counter += 1
    elif (cv2.waitKey(20) & 0xff) == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
