from mtcnn import MTCNN
import cv2,os

data_path='dataset_trial'

img_size=100

folder_path=os.path.join(data_path,"without_mask")
img_names=os.listdir(folder_path)
detector = MTCNN()

for img_name in img_names:
    img_path=os.path.join(folder_path,img_name)
    img=cv2.imread(img_path)
    
    try:
        result= detector.detect_faces(img)
        if result!=[]:
            for person in result:
                bounding_box=person['box']
                keypoints=person['keypoints']
                cv2.rectangle(img,(bounding_box[0], bounding_box[1]),(bounding_box[0]+bounding_box[2], bounding_box[1] + bounding_box[3]),(0,155,255),2)
                img=img[bounding_box[1]:bounding_box[1]+bounding_box[3],bounding_box[0]:bounding_box[0]+bounding_box[2]]
                img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                resized=cv2.resize(img,(img_size,img_size))
                target_dir = r'dataset_trial\unmask'
                img_path=os.path.join(target_dir,img_name)
                cv2.imwrite(img_path, resized)
    except Exception as e:
        print('Exception:',e)

folder_path=os.path.join(data_path,"with_mask")
img_names=os.listdir(folder_path)

for img_name in img_names:
    img_path=os.path.join(folder_path,img_name)
    img=cv2.imread(img_path)
    
    try:
        result= detector.detect_faces(img)
        if result!=[]:
            for person in result:
                bounding_box=person['box']
                keypoints=person['keypoints']
                cv2.rectangle(img,(bounding_box[0], bounding_box[1]),(bounding_box[0]+bounding_box[2], bounding_box[1] + bounding_box[3]),(0,155,255),2)
                img=img[bounding_box[1]:bounding_box[1]+bounding_box[3],bounding_box[0]:bounding_box[0]+bounding_box[2]]
                img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                resized=cv2.resize(img,(img_size,img_size))
                target_dir = r'dataset_trial\mask'
                img_path=os.path.join(target_dir,img_name)
                cv2.imwrite(img_path, resized)
    except Exception as e:
        print('Exception:',e)
