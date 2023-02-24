from flask import Flask, request, jsonify
from joblib import load
import cv2
import numpy as np
import os

def cropWhiteArea(img):
        #Get the original width and height of the image
        heightOriginal, widthOriginal, channels = img.shape
        # Convert the image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Threshold the image to get a binary mask of the white pixels
        thresh = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)[1]

        # Find the coordinates of the non-white pixels in the binary mask
        nonwhite_coords = np.where(thresh != 255)

        # Compute the bounding box of the non-white pixels
        cropMargin = 0.05
        x_min = np.min(nonwhite_coords[1])
        y_min = np.min(nonwhite_coords[0])
        x_max = np.max(nonwhite_coords[1])
        y_max = np.max(nonwhite_coords[0])
        H = y_max - y_min
        W = x_max - x_min
        
        x_min = round(max(x_min-cropMargin*W,0))
        y_min = round(max(y_min-cropMargin*H,0))
        x_max = round(min(x_max+cropMargin*W,widthOriginal))
        y_max = round(min(y_max+cropMargin*H,heightOriginal))
        

        # Crop the image to the bounding box
        cropped_img = img[y_min:y_max, x_min:x_max]
        return cropped_img

def makeImageSquare(img):
        # Get the original width and height of the image
        height, width, channels = img.shape

        # Find the maximum dimension of the image
        max_dim = max(height, width)

        # Compute the amount of padding needed to make the image square
        h_padding = (max_dim - height) // 2
        w_padding = (max_dim - width) // 2

        # Pad the image with white pixels to make it square
        padded_img = cv2.copyMakeBorder(img, h_padding, h_padding, w_padding, w_padding, cv2.BORDER_CONSTANT, value=[255, 255, 255])
        return padded_img
  
def preProcess(img):
          
    # Define the image size for the deep learning model
    target_width, target_height = 48, 48

    # Define the threshold for the amount of white pixels to consider an image as having a large white area
    white_threshold = 0.9
    
    img_crop = cropWhiteArea(img)
    imgSquare = makeImageSquare(img_crop)
    resized_img = cv2.resize(imgSquare, (target_width, target_height))
    
    # Convert the image to grayscale
    gray = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
    
    return gray
    
def showImages(img1, img2):
    ## Create a new figure and a grid of subplots with 1 row and 2 columns
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

    # Display the first image on the first subplot
    axes[0].imshow(img)

    # Display the second image on the second subplot
    axes[1].imshow(img_modified)

    # Show the plot
    plt.show()
    return 0

def predictClass(givenImg):
    img_modified = preProcess(givenImg)
    Kcomponents = 70
    image_matrix = img_modified.reshape(1, -1)
    SVD_inverseMat = np.load("SVD_inverse.npy")
    features = image_matrix @ SVD_inverseMat
    features = features[:, :Kcomponents]
    # Load the saved model from disk
    clf = load('DT_model.joblib')   
    y_pred = clf.predict(features)
    return y_pred[0]



app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    # Get the file attachment from the request
    file = request.files['image']
    
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_UNCHANGED)

    predictedLabel = predictClass(img)

    # Return the predicted label as a JSON response
    return jsonify({'label': predictedLabel})

if __name__ == '__main__':
    app.run(debug=True)