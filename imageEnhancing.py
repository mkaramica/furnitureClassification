import cv2
import numpy as np
import os


originalPath = "Original"

classNames = ["Bed","Chair","Sofa"]

destination = "ModifiedPics"


def makeDirs():
    if not os.path.exists(destination):
        os.mkdir(folder)
    for className in classNames:
        dest_path = os.path.join(destination, className)
        if not os.path.exists(dest_path):
            os.mkdir(dest_path)   
    return 0

makeDirs()

# Define the image size for the deep learning model
target_width, target_height = 48, 48

# Define the threshold for the amount of white pixels to consider an image as having a large white area
white_threshold = 0.9

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


# Loop over the image files in each directory and resize and save the images to the appropriate subdirectory
n=1
for imageClass in classNames:
    imagePath = os.path.join(originalPath, imageClass)
    for filename in os.listdir(imagePath):
        if filename.endswith('.jpg'):
            # Read the image using OpenCV
            img = cv2.imread(os.path.join(imagePath, filename))
            img_crop = cropWhiteArea(img)
            imgSquare = makeImageSquare(img_crop)
            resized_img = cv2.resize(imgSquare, (target_width, target_height))
            
            # Convert the image to grayscale
            gray = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)

            # Perform histogram equalization on the grayscale image
            equalized = cv2.equalizeHist(gray)
       
       
            dest_Gray = os.path.join(destination, imageClass, filename)
            cv2.imwrite(dest_Gray, gray)

            
            print(n,filename)
            n=n+1