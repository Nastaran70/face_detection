# Face Detection Using OpenCV

This is a simple code for face detection using the OpenCV DNN Face Detector. I have used `cv2.dnn.readNetFromTensorflow` since I was using 8 bit quantized Tensorflow version.
The file opencv_face_detection.py will prompt the user to give the path to an image for detecting faces.
The file cam_opencv_face_detection.py will use a camera to capture an image and detects faces. Use the "q" key to capture image. press any key to proceed to detection. Tested and worked on laptop camera.
Both will leave the original image window open and show the bonding box in a new window. press any key to proceed to detection.
Both will save the image with bonding boxes tn the same directory. The camera version also saves the original image captured from the camera.
The model and config file are included in `./Models` . If you are running this code, please either clone the repo, or make sure to download the models as well, and edit the path to files accordingly.
Code has been writen and tested using `Python 3.10.13` and `OpenCV 4.10.0` .
