import cv2
from tensorflow.keras.models import load_model
from mtcnn import MTCNN
import numpy as np

detector = MTCNN()
source=cv2.VideoCapture(0)
model = load_model('model-018.model')
labels_dict={0:'MASK',1:'NO MASK'}

while True: 
    _,img=source.read()
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    result = detector.detect_faces(img)
    if result != []:
        for person in result:
            bounding_box = person['box']
            keypoints = person['keypoints']
            face_img=gray[bounding_box[1]:bounding_box[1]+bounding_box[2],bounding_box[0]:bounding_box[0]+bounding_box[2]]
            resized=cv2.resize(face_img,(100,100))
            normalized=resized/255.0
            reshaped=np.reshape(normalized,(1,100,100,1))
            predicted=model.predict(reshaped)

            label=np.argmax(predicted,axis=1)[0]
    
            cv2.rectangle(img,
                          (bounding_box[0], bounding_box[1]),
                          (bounding_box[0]+bounding_box[2], bounding_box[1] + bounding_box[3]),
                          (0,155,255),
                          2)
            cv2.rectangle(img,
                          (bounding_box[0], bounding_box[1]-40),
                          (bounding_box[0]+bounding_box[2], bounding_box[1]),
                          (0,155,255),
                          -1)
            cv2.putText(img, labels_dict[label], (bounding_box[0], bounding_box[1]-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
            acc=round(np.max(predicted,axis=1)[0]*100,2)
            cv2.putText(img,str(acc),(bounding_box[0]+150,bounding_box[1]-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
            cv2.circle(img,(keypoints['left_eye']), 2, (0,155,255), 2)
            cv2.circle(img,(keypoints['right_eye']), 2, (0,155,255), 2)
            cv2.circle(img,(keypoints['nose']), 2, (0,155,255), 2)
            cv2.circle(img,(keypoints['mouth_left']), 2, (0,155,255), 2)
            cv2.circle(img,(keypoints['mouth_right']), 2, (0,155,255), 2)
    cv2.imshow('LIVE',img)
    key=cv2.waitKey(1)
    
    if(key==27):
        break
        
cv2.destroyAllWindows()
source.release()
