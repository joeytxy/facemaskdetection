import os
import time

import matplotlib as mpl
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

from PIL import Image

from skimage.feature import hog
from skimage.color import rgb2gray

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC

from sklearn.metrics import roc_curve, auc
import pickle

start = time.time()
mask_train_folder_path = "data/mask_train"
unmask_train_folder_path = "data/unmask_train"

mask_img_names = os.listdir(mask_train_folder_path)
unmask_img_names = os.listdir(unmask_train_folder_path)

allimages_path = []
for i in mask_img_names:
    img_path = os.path.join(mask_train_folder_path, i)
    allimages_path.append(img_path)
    
for i in unmask_img_names:
    img_path = os.path.join(unmask_train_folder_path, i)
    allimages_path.append(img_path)
    
target_mask_train = [1]*len(mask_img_names)
target_unmask_train = [0]*len(unmask_img_names)
y_train = np.concatenate((target_mask_train, target_unmask_train))

mask_test_folder_path = "data/mask_test"
unmask_test_folder_path = "data/unmask_test"

mask_img_names = os.listdir(mask_test_folder_path)
unmask_img_names = os.listdir(unmask_test_folder_path)

alltestimages_path = []
for i in mask_img_names:
    img_path = os.path.join(mask_test_folder_path, i)
    alltestimages_path.append(img_path)
    
for i in unmask_img_names:
    img_path = os.path.join(unmask_test_folder_path, i)
    alltestimages_path.append(img_path)
    
target_mask_test = [1]*len(mask_img_names)
target_unmask_test = [0]*len(unmask_img_names)
y_test = np.concatenate((target_mask_test, target_unmask_test))

# from the file names in the folder, open the image using the object in Pillow, and
# then return the image as a numpy array

def get_image(i):
    img = Image.open(i)
    return np.array(img)

def create_features(img):
    #flatten three channel colour image
    color_features = img.flatten()
    
    # convert image to greyscale
    grey_image = rgb2gray(img)
    
    # get HOG features from greyscale image
    hog_features = hog(grey_image, block_norm = 'L2-Hys', pixels_per_cell = (16, 16))

    # combine colour and hog features into a single array
    flat_features = np.hstack(color_features)
    return flat_features

def create_feature_matrix(paths):
    features_list = []
    for img_id in paths:
        #load image
        img = get_image(img_id)
        #get features for image
        image_features = create_features(img)
        features_list.append(image_features)

    feature_matrix = np.array(features_list)
    return feature_matrix

feature_matrix = create_feature_matrix(allimages_path)
feature_matrix_test = create_feature_matrix(alltestimages_path)

print('Feature matrix shape is: ', feature_matrix.shape)
ss = StandardScaler()
stand = ss.fit_transform(feature_matrix)
pca = PCA(n_components = len(allimages_path))
masks_pca = ss.fit_transform(stand)
print('PCA matrix is: ', masks_pca.shape)

X_train = pd.DataFrame(masks_pca)

print('Feature matrix shape is: ', feature_matrix_test.shape)
ss = StandardScaler()
stand = ss.fit_transform(feature_matrix_test)
pca = PCA(n_components = len(alltestimages_path))
test_masks_pca = ss.fit_transform(stand)
print('PCA matrix is: ', test_masks_pca.shape)

X_test = pd.DataFrame(test_masks_pca)

#test on training data
#X_test = pd.DataFrame(masks_pca)

# define support vector classifier
svm = SVC(kernel='rbf')

# fit model
svm.fit(X_train, y_train)
filename="svm_model.sav"
pickle.dump(svm,open(filename,'wb'))
# generate predictions
y_pred = svm.predict(X_test)

#test on training data
#accuracy = accuracy_score(y_train, y_pred)

# calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print('Model accuracy is: ', accuracy)

stop = time.time()
print(f"Training time: {stop - start}s")
