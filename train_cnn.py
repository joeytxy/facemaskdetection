import cv2,os
import numpy as np
from tensorflow.keras.models import load_model
from keras.utils import np_utils

data=[]
target=[]
img_size=100

# Category 0 is mask
categories=os.listdir("3244")
#labels=[i for i in range(len(categories))]

folder_path=r"3244\mask_train"
img_names=os.listdir(folder_path)
    
for img_name in img_names:
    img_path=os.path.join(folder_path,img_name)
    img=cv2.imread(img_path)
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)  
    data.append(img)
    target.append(0)

folder_path=r"3244\unmask_train"
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

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Activation,MaxPooling2D, Dropout
from keras.callbacks import ModelCheckpoint

model=Sequential()

model.add(Conv2D(200,(3,3),input_shape=data.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
#The first CNN layer followed by Relu and MaxPooling layers

model.add(Conv2D(100,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
#The second convolution layer followed by Relu and MaxPooling layers


model.add(Flatten())
model.add(Dropout(0.5))
#Flatten layer to stack the output convolutions from second convolution layer
model.add(Dense(50,activation='relu'))
#Dense layer of 64 neurons
model.add(Dense(2,activation='softmax'))
#The Final layer with two outputs for two categories

#If your input images are greater than 128×128 you may choose to use a kernel size > 3 to help (1) learn larger spatial filters and (2) to help reduce volume size.

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

from sklearn.model_selection import train_test_split

train_data,test_data,train_target,test_target=train_test_split(data,target,test_size=0.1)
checkpoint = ModelCheckpoint('model-{epoch:03d}.model',monitor='val_loss',verbose=0,save_best_only=True,mode='auto')
history=model.fit(train_data,train_target,epochs=20,callbacks=[checkpoint],validation_split=0.2)

#from matplotlib import pyplot as plt

#plt.plot(history.history['loss'],'r',label='training loss')
#plt.plot(history.history['val_loss'],label='validation loss')
#plt.xlabel('# epochs')
#plt.ylabel('loss')
#plt.legend()
#plt.show()

#plt.plot(history.history['accuracy'],'r',label='training accuracy')
#plt.plot(history.history['val_accuracy'],label='validation accuracy')
#plt.xlabel('# epochs')
#plt.ylabel('loss')
#plt.legend()
#plt.show()

#print(model.evaluate(test_data,test_target))
