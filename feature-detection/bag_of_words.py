import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imread
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import filters

# NOTE:
# There has been a TODO put in all of the places you need to add code


folder = './images/'
categories = ['tractor', 'coral reef', 'banana', 'fountain']


def image_patches(im, patch_size=(16,16)):
    # Returns image patches from the image  
    # Input- im: H x W x C
    #        patch_size: U X V 
    # Output- patches: 16 x 768 
    patches = np.zeros((16,768))
    temp = filters.image_patches(im, patch_size)
    
    for i in range(len(temp)):
        patches[i,:] = temp[i].reshape(1,-1)
    # TODO Extract patches from an image
    return patches


def build_codebook(X_Train, num_clusters=15):
    # Returns a KMeans object fit to the dataset  
    # Input- X_train: (3*N/4 * M) x P
    #        num_clusters: scalar 
    # Output- KMeans: object
    codebook = KMeans(n_clusters=num_clusters).fit(X_Train)
    return codebook


def normalize_and_split(X, y):
    # Returns the normalized, split dataset in patches
    # N = num of samples
    # M = num of patches per image
    # P = size of patch flattened to a vector
    # Input- X: N x M x P 
    #        y: N x 1
    # Output- X_train: (3*N/4 * M) x P
    #         X_test: (N/4 * M) x 1
    #         y_train: 3*N/4 x 1
    #         y_test: N/4 x 1
    X = np.asarray(X)
    patch = X[0] # Need to fetch the patch size.. should be (16, 768)

    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=0.25, random_state=42)

    # Shape it so it is of size (Num of Patches x Size of Patch
    X_train = X_train.reshape(-1, patch.shape[1])
    X_test = X_test.reshape(-1, patch.shape[1])

    # After building X, it must be normalized.
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test


def train_classifier(X_hist_train, X_hist_test, y_train, y_test):
    # Trains a classifier and evaluates it
    # Input- X_hist_train: 1500 x 15 
    #        X_hist_test: 500 x 15
    #        y_train: 1500 x 1
    #        y_test: 500 x 1
    # Output- clf: classifier object 
    #         score: scalar 
    clf = SVC()

    clf.fit(X_hist_train, y_train)
    score = clf.score(X_hist_test, y_test)
    print("Validation Performance: {}".format(str(score)))
    return clf, score


def main():
    X = []
    y = []
    paths = []

    # 1. TODO Iterate over the images in `images/`
    # Example with only bananas (you have to do for all 4 classes): 
    paths1 = []
    paths2 = []
    paths3 = []
    paths += ['./images/banana/' + f for f in os.listdir('./images/banana/')] 
    paths1 += ['./images/coral_reef/' + f for f in os.listdir('./images/coral_reef/')] 
    paths2 += ['./images/fountain/' + f for f in os.listdir('./images/fountain/')] 
    paths3 += ['./images/tractor/' + f for f in os.listdir('./images/tractor/')] 
    

    # 2. TODO Extract the patches from each of them and them to X and the class labels to y
    for p in paths:
        patches = np.zeros((16,768))
        # TODO load color image
        # TODO get image_patches using above function you write
        img = cv2.imread(p)
        patches = image_patches(img)
        X.append(patches)
        y.append('0')
    for p in paths1:
        patches = np.zeros((16,768))
        # TODO load color image
        # TODO get image_patches using above function you write
        img = cv2.imread(p)
        patches = image_patches(img)
        X.append(patches)
        y.append('1')
    for p in paths2:
        patches = np.zeros((16,768))
        # TODO load color image
        # TODO get image_patches using above function you write
        img = cv2.imread(p)
        patches = image_patches(img)
        X.append(patches)
        y.append('2')
    for p in paths3:
        patches = np.zeros((16,768))
        # TODO load color image
        # TODO get image_patches using above function you write
        img = cv2.imread(p)
        patches = image_patches(img)
        X.append(patches)
        y.append('3')
    
    
    #plt.imshow(X[0][10,:].reshape(16,16,3))
    #plt.show()
    
    # 4. Here the code to normalize and split your patches has been written already
    X_train, X_test, y_train, y_test = normalize_and_split(X, y)

    # 5. Build the codebook with X_train
    codebook = build_codebook(X_train)

    # 6. Produce the labels for each of the test samples
    labels = codebook.predict(X_test)

    # 7. Reshape to appropriate sizes 
    X_train = X_train.reshape(1500, 16, 768)
    X_test = X_test.reshape(500, 16, 768)

    # 8. TODO Build your histogram
    X_hist_train = np.zeros((1500, 15))
    X_hist_test = np.zeros((500, 15))
    for i in range(X_train.shape[0]):    
        temp_labels = codebook.predict(X_train[i])
        for j in temp_labels:
            X_hist_train[i,j] += 1
    for i in range(X_test.shape[0]):
        temp_labels = codebook.predict(X_test[i])
        for j in temp_labels:
            X_hist_test[i,j] += 1
    
    
    # 9. Train the classifier
    train_classifier(X_hist_train, X_hist_test, y_train, y_test)


if __name__ == "__main__":
    main()
