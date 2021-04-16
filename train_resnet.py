import cv2,os
import numpy as np
from keras.utils import np_utils

data=[]
target=[]
img_size=100

# Category 0 is mask
categories=os.listdir("data")

folder_path=r"data\mask_train"
img_names=os.listdir(folder_path)
    
for img_name in img_names:
    img_path=os.path.join(folder_path,img_name)
    img=cv2.imread(img_path)
    # img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)  
    data.append(img)
    target.append(0)

folder_path=r"data\unmask_train"
img_names=os.listdir(folder_path)
    
for img_name in img_names:
    img_path=os.path.join(folder_path,img_name)
    img=cv2.imread(img_path)
    # img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)  
    data.append(img)
    target.append(1)
        
data=np.array(data)/255.0
data=np.reshape(data,(data.shape[0],img_size,img_size,3))
target=np.array(target)
target=np_utils.to_categorical(target)

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Activation,MaxPooling2D, Dropout
from keras.callbacks import ModelCheckpoint
from keras.applications import ResNet50
from keras.applications.resnet50 import preprocess_input

model = Sequential()
model.add(ResNet50(include_top = False, weights = "imagenet", pooling = "avg", input_shape = (100, 100, 3)))
model.add(Dense(50,activation='relu'))
model.add(Dense(2,activation='softmax'))
model.layers[0].trainable = False
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])


from sklearn.model_selection import train_test_split

train_data,test_data,train_target,test_target=train_test_split(data,target,test_size=0.1)
checkpoint = ModelCheckpoint('model-{epoch:03d}.model',monitor='val_loss',verbose=0,save_best_only=True,mode='auto')
history=model.fit(train_data,train_target,epochs=600,callbacks=[checkpoint],validation_split=0.2)

from matplotlib import pyplot as plt

plt.plot(history.history['accuracy'],'r',label='training accuracy')
plt.plot(history.history['val_accuracy'],label='validation accuracy')
plt.xlabel('# epochs')
plt.ylabel('loss')
plt.legend()
plt.show()
