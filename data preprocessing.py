import os
import mtcnn
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import PIL.ImageOps

data_path='data'
folder_path=os.path.join(data_path,"unmask")
img_names=os.listdir(folder_path)
detector = mtcnn.MTCNN()

img_num = 0
for img_name in img_names:
    pixels = plt.imread(os.path.join(folder_path, img_name))
    img = cv2.imread(os.path.join(folder_path, img_name))
    try:
        faces = detector.detect_faces(pixels)
        print(img_num, img_name, len(faces))
        count = 0
        for face in faces:
            x, y, width, height = face['box']
            img = img[y:y+height, x:x+width]
            if img.size == 0:
                continue
            if len(img.shape) != 3:
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (100,100))
            new_image_name = os.path.join(r'data\unmask_train', img_name + str(count) + '.jpg')
            cv2.imwrite(new_image_name, gray)
            count = count + 1
        img_num = img_num + 1
    except Exception as e:
        print('Exception:',e)
