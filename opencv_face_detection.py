import cv2

def detect_faces(image, net):
  
    height, width, channels = image.shape

    ## convert image to a blob and pass through the network:
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), [104, 117, 123], False, False)  
    net.setInput(blob) 
    detections = net.forward() 

    faces = []

    for i in range(detections.shape[2]):    ## collect faces and Bboxes
        confidence = detections[0, 0, i, 2]
        if confidence > 0.75:    ## only faces with a confidence equal or higher than 75% are considered. adjust if needed.
            x1 = int(detections[0, 0, i, 3] * width)
            y1 = int(detections[0, 0, i, 4] * height)
            x2 = int(detections[0, 0, i, 5] * width)
            y2 = int(detections[0, 0, i, 6] * height)
            faces.append([x1, y1, x2, y2])

    return faces


def draw_facebox(image, result_list):

    ## draw each bounding box
    for result in result_list:
        ## get coordinates
        x1, y1, x2, y2 = result
        ## draw the rectangle on the image
        cv2.rectangle(image, (x1, y1), (x2,y2), (0, 255, 0), 2)

    ## display the image with rectangles
    cv2.imshow("Faces detected!", image)
    cv2.waitKey()
    cv2.imwrite("image_with_bbox.jpg", image) ## saving the image with Bboxes for future reference




## loading opencv face detector model:
modelFile = "./Models/opencv_face_detector_uint8.pb"
configFile = "./Models/opencv_face_detector.pbtxt"
net = cv2.dnn.readNetFromTensorflow(modelFile, configFile)

img_path = input("Please enter the location of the image file: ")    ## prompt the user to input the file path
image = cv2.imread(img_path)   ## read the image
cv2.imshow("Original Image", image)  ## show the original image
cv2.waitKey()
results = detect_faces(image, net)  ## detect faces
draw_facebox(image, results)  ## draw and show image with faceboxes 
