from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

import numpy as np

import os
from imutils.video import VideoStream 
import imutils

def detect_and_predict_mask(frame,faceNet,maskNet):
    #grab the dimensions of the frame and then construct a blob
    (h,w)=frame.shape[:2]
    blob=cv2.dnn.blobFromImage(frame,1.0,(300,300),(104.0,177.0,123.0))
    
    faceNet.setInput(blob)
    detections=faceNet.forward()
    
    #initialize our list of faces, their corresponding locations and list of predictions
    
    faces=[]
    locs=[]
    preds=[]
    
    
    for i in range(0,detections.shape[2]):
        confidence=detections[0,0,i,2]
    
    
        if confidence>0.5:
        #we need the X,Y coordinates
            box=detections[0,0,i,3:7]*np.array([w,h,w,h])
            (startX,startY,endX,endY)=box.astype('int')

            #ensure the bounding boxes fall within the dimensions of the frame
            (startX,startY)=(max(0,startX),max(0,startY))
            (endX,endY)=(min(w-1,endX), min(h-1,endY))

            #extract the face ROI, convert it from BGR to RGB channel, resize it to 224,224 and preprocess it
            face=frame[startY:endY, startX:endX]
            face=cv2.cvtColor(face,cv2.COLOR_BGR2RGB)
            face=cv2.resize(face,(100,100))
            face=img_to_array(face)
            face=preprocess_input(face)

            faces.append(face)
            locs.append((startX,startY,endX,endY))
        
        #only make a predictions if atleast one face was detected
        if len(faces)>0:
            faces=np.array(faces,dtype='float32')
            preds=maskNet.predict(faces,batch_size=32)
            
            
        return (locs,preds)
    
    

faceNet=cv2.dnn.readNet('deploy.prototxt','res10_300x300_ssd_iter_140000.caffemodel')
maskNet=load_model('best.keras')


vs=VideoStream(src=1 ).start()



def video_stream():
    while True:
        #grab the frame from the threaded video stream and resize it
        #to have a maximum width of 400 pixels
        frame=vs.read()
        
        #frame=imutils.resize(frame,width=400)

        #detect faces in the frame and preict if they are waring masks or not
        (locs,preds)=detect_and_predict_mask(frame,faceNet,maskNet)



        #loop over the detected face locations and their corrosponding loactions
        #print(type(preds))
        for (box, pred) in zip(locs, preds):
            (startX,startY,endX,endY)=box

            (NO_MASK,INCORRECT,MASK) = pred
            label="NO MASK"
            color=(0,0,255)
            if NO_MASK<MASK:
                if MASK<INCORRECT:
                    label="INCORRECT"
                    color=(255,0,0)

                else :
                    label="Mask"
                    color=(0,255,0)
            #determine the class label and color we will use to draw the bounding box and text



            #display the label and bounding boxes
            mask_label = "{}: {:.2f}%".format(label, max(NO_MASK,INCORRECT,MASK) * 100)
            cv2.putText(frame,mask_label,(startX,startY-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,color,2)
            cv2.rectangle(frame,(startX,startY),(endX,endY),color,2)

        #show the output frame
        
        ret,buffer=cv2.imencode('.jpg',frame)
        frame=buffer.tobytes()

        yield(b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
