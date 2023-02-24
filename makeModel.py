import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split, GridSearchCV
from joblib import dump


imageFolder = 'ModifiedPics'

classNames = ["Bed","Chair","Sofa"]

Kcomponents = 70

def makeClassDict():
    class_dict = {}
    # Traverse the directory structure and store the class for each image in the dictionary
    for subdir, dirs, files in os.walk(imageFolder):
        for file in files:
            if file.endswith('.jpg'):
                class_name = os.path.basename(subdir)
                image_path = os.path.join(subdir, file)
                class_dict[image_path] = class_name
    return class_dict

def shuffleImages(class_dict):
    # Get a list of image names and shuffle them
    image_names = list(class_dict.keys())
    np.random.shuffle(image_names)

    # Get a list of image names and shuffle them
    image_names = list(class_dict.keys())
    np.random.shuffle(image_names)
    return image_names

def reconstructImages(uSVD, sSVD, vSVD, Kcomponents, n,h,w):
    # Reconstruct the matrix using the SVD components
    reconstructed_matrix = uSVD[:, :Kcomponents] @ np.diag(sSVD[:Kcomponents]) @ vSVD[:Kcomponents, :]

    # Reshape the matrix into the original image shape
    reconstructed_images = reconstructed_matrix.reshape(n, h, w)
    
    return reconstructed_images


def showRandomImages(images,reconstructed_images, nShow ):
    for iImg in range(0,nShow):
        # Create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2)

        # Display the images in the subplots
        ax1.imshow(images[iImg])
        ax2.imshow(reconstructed_images[iImg])

        # Set the titles for the subplots
        ax1.set_title('Image Original')
        ax2.set_title('Image Reconstructed')

        # Show the plot
        plt.show()
        
    return 0
    
    
class_dict = makeClassDict()
image_names = shuffleImages(class_dict)

images = []
# Loop through the shuffled image names and retrieve the class for each image
for name in image_names:
    class_name = class_dict[name]
    img = cv2.imread(name,cv2.IMREAD_GRAYSCALE)
    images.append(img)

# Stack the images into a 3D array
image_array = np.stack(images)
n, h, w = image_array.shape

# Flatten the array into a 2D matrix
image_matrix = image_array.reshape(n, -1)

# Perform SVD on the matrix
uSVD, sSVD, vSVD = np.linalg.svd(image_matrix)

SVD_inverseMat = np.linalg.pinv(np.diag(sSVD[:Kcomponents]) @ vSVD[:Kcomponents, :])
#Save SVD inverse Matrix:
np.save('SVD_inverse.npy', SVD_inverseMat)


features = uSVD[:, :Kcomponents]
 
labels =[]
for name in image_names:
    class_name = class_dict[name]
    labels.append(class_name)


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)


# Create a decision tree classifier
clf = DecisionTreeClassifier()

# Define the hyperparameter search space
param_grid = {
    'max_depth': [2, 4, 6, 8, 10],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 4, 8]
}

# Perform a grid search over the hyperparameter space
grid_search = GridSearchCV(clf, param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Print the best hyperparameters and the testing accuracy
print(f'Best hyperparameters: {grid_search.best_params_}')
print(f'Testing accuracy: {grid_search.score(X_test, y_test):.3f}')

# Train a new model with the best hyperparameters on the full inner training set
clf_best = DecisionTreeClassifier(**grid_search.best_params_)
clf_best.fit(features, labels)

# Save the trained model to disk
dump(clf_best, 'DT_model.joblib')

# Plot the decision tree
plt.figure(figsize=(15, 10))
plot_tree(clf_best, filled=True)
plt.show()
