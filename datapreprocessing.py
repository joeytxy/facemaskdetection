import cv2,os
import numpy as np
from tensorflow.keras.models import load_model
from keras.utils import np_utils

data_path='data'

img_size=100

folder_path=os.path.join(data_path,"without_mask")
img_names=os.listdir(folder_path)
    
for img_name in img_names:
    img_path=os.path.join(folder_path,img_name)
    img=cv2.imread(img_path)
    
    try:
        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        resized=cv2.resize(gray,(img_size,img_size))
        target_dir = r'C:\Users\joeyt\Desktop\3244\unmasked'
        img_path=os.path.join(target_dir,img_name)
        cv2.imwrite(img_path, resized)
    except Exception as e:
        print('Exception:',e)


