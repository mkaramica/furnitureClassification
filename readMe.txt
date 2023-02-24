1- The given database is to be placed inside the "Original" folder.
2- Run Image Enhancing to pre-process the images, including crop, resizing, and making them gray-scale.
This will create another folder called "ModifiedPics" to store the resized images.
2- Run Make Model to build the predicting model. It builds the predicting model and stores it in the root of the project folder. 
Two files will be created:
	2-1 DT_model.joblib: The decision tree mode
	2-2 SVD_inverse.npy: The matrix is used to perform inverse SVD operation on the given image
3- Run UseTrainedModel to use the built model for predicting the class of any image, existing on the computer 
4- Run app.py, which is a api-enabled app to receive image by api and return its predicted label.
5- The model accuracy varies according to random selection of train/test sets, but it is generally in range (0.83 to 0.95).