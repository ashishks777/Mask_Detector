from subprocess import STDOUT, check_call
proc = subprocess.Popen('apt-get install -y libgl1-mesa-glx', shell=True, stdin=None, stdout=open(os.devnull,"wb"), stderr=STDOUT, executable="/bin/bash")
proc.wait()
from tensorflow.keras.models import load_model

import numpy as np
import cv2

haar_data=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")


capture=cv2.VideoCapture(1)
def video_stream():
    while True:
        flag,frame =capture.read()
        if flag:
            faces=haar_data.detectMultiScale(frame)
            for x,y,w,h in faces:
                cv2.rectangle(frame,(x,y),(x+w,y+h),(120,175,159),4)
                face=frame[y:y+h,x:x+w,:]
                face=cv2.resize(face,(100,100))
                face=face.reshape(1,100,100,3)
                face=face/255
                pr=model.predict(face)
                cv2.putText(frame,mask_label[pr.argmax()],(x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),2)

        ret,buffer=cv2.imencode('.jpg',frame)
        frame=buffer.tobytes()

        yield(b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
