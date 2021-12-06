import glob
import os
import cv2
import natsort
from sklearn import decomposition
from sklearn import datasets
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import decomposition
import numpy as np

# load dataset
# files_folder = '/home/azureuser/hackathon_data/hackathon_data/1634698996.767572/processed_images/0'
files_folder = '/home/azureuser/hackathon_data/hackathon_data'
img_list = natsort.natsorted(glob.glob(os.path.join(files_folder, '*/processed_images/*/*.png')))
total_n_imgs = len(img_list)
print("Total number of images to be processed = {}".format(total_n_imgs))

img_tmp = cv2.imread(img_list[0])
# plt.imshow(img_tmp, cmap='gray')
height, width, layers = img_tmp.shape

img_data = np.zeros(shape=(total_n_imgs, height, width))
targets_data = np.zeros(shape=(total_n_imgs))
for i, filename in enumerate(img_list):
    if i%100 == 0:
        print("Reading files {} ou of {}, or {}%".format(i, total_n_imgs, int(100.0*i/total_n_imgs)))
    img_data[i] = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
print("Finished reading")

fig = plt.figure(figsize=(8, 6))
# plot several images
for i in range(15):
    ax = fig.add_subplot(3, 5, i + 1, xticks=[], yticks=[])
    ax.imshow(img_data[i], cmap='gray')
plt.show()

# flatten images
img_data_flat = np.reshape(img_data, newshape=(total_n_imgs, height*width))

X_train, X_test, y_train, y_test = train_test_split(img_data_flat, targets_data, train_size=0.9, random_state=0, shuffle=True)

print(X_train.shape, X_test.shape)

print("Started PCA")
pca = decomposition.PCA(n_components=150, whiten=True)
pca.fit(X_train)
print("Finished PCA")

plt.imshow(pca.mean_.reshape(img_data[0].shape), cmap='gray')
plt.show()

print(pca.components_.shape)

fig = plt.figure(figsize=(16, 6))
for i in range(30):
    ax = fig.add_subplot(3, 10, i + 1, xticks=[], yticks=[])
    ax.imshow(pca.components_[i].reshape(img_data[0].shape), cmap='gray')
plt.show()

X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
print(X_train_pca.shape)

print(X_test_pca.shape)

variance = pca.explained_variance_ratio_
var=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=3)*100)
var #cumulative sum of variance explained with [n] features
plt.ylabel('% Variance Explained')
plt.xlabel('# of Features')
plt.title('PCA Analysis')
# plt.ylim(30,100.5)
plt.style.context('seaborn-whitegrid')
plt.plot(var)
plt.show()