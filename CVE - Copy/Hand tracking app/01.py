# Touch free interaction
# Sign Language Recognition , Gaming (fist means to attack), virtual lab

import cv2 # Import package

webcam = cv2.VideoCapture(0) # Create webcam object

while True:
    response, frame = webcam.read()
    cv2.imshow("WEBCAM", frame)
    cv2.waitKey(1)
    
#print(frame) #in np array