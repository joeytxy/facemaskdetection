import cv2,os
from tensorflow.keras.models import load_model
import numpy as np

model = load_model('model-018.model')


folder_path=r"3244\mask_test"
img_names=os.listdir(folder_path)
count=0
correct=0
    
for img_name in img_names:
    count=count+1
    img_path=os.path.join(folder_path,img_name)
    img=cv2.imread(img_path)
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img=img/255.0
    img=np.reshape(img,(1,100,100,1))
    result=model.predict(img)
    if (result[0][0] > result[0][1]):
        correct=correct+1

folder_path=r"3244\unmask_test"
img_names=os.listdir(folder_path)
    
for img_name in img_names:
    count=count+1
    img_path=os.path.join(folder_path,img_name)
    img=cv2.imread(img_path)
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img=img/255.0
    img=np.reshape(img,(1,100,100,1))
    result=model.predict(img)
    if (result[0][0] < result[0][1]):
        correct=correct+1
print(correct/count)
        
