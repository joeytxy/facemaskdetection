import cv2,os
import numpy as np
from keras.utils import np_utils
from matplotlib import pyplot as plt
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Activation,MaxPooling2D, Dropout
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split

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
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)  
    data.append(img)
    target.append(0)

folder_path=r"data\unmask_train"
img_names=os.listdir(folder_path)
    
for img_name in img_names:
    img_path=os.path.join(folder_path,img_name)
    img=cv2.imread(img_path)
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)  
    data.append(img)
    target.append(1)
        
data=np.array(data)/255.0
data=np.reshape(data,(data.shape[0],img_size,img_size,1))
target=np.array(target)
target=np_utils.to_categorical(target)


model=Sequential()

model.add(Conv2D(200,(3,3),activation='relu',input_shape=data.shape[1:]))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(100,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(100,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(50,activation='relu'))
model.add(Dense(2,activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

train_data,test_data,train_target,test_target=train_test_split(data,target,test_size=0.1)
checkpoint = ModelCheckpoint('model-{epoch:03d}.model',monitor='val_loss',verbose=0,save_best_only=True,mode='auto')
history=model.fit(train_data,train_target,epochs=20,callbacks=[checkpoint],validation_split=0.2)

plt.plot(history.history['accuracy'],'r',label='training accuracy')
plt.plot(history.history['val_accuracy'],label='validation accuracy')
plt.xlabel('# epochs')
plt.ylabel('loss')
plt.legend()
plt.show()

