import cv2
from hand_tracking import HandDetector

webcam = cv2.VideoCapture(0) # Set the correct camera / webcam.
webcam.set(3, 1_280) # Set resolution width
webcam.set(4, 720) # Set resolution height

detector = HandDetector(min_detection_confidence = 0.8)

shape_color = color_purple = (255, 0, 255)
color_green = (0, 255, 0)

shape_centre_x = 100
shape_centre_y = 100
shape_width = 200
shape_height = 200

while True:
    response, frame = webcam.read()
    frame = cv2.flip(frame, 1) # Flip the webcam output
    frame = detector.find_hands(frame = frame)
    list_landmarks = detector.find_position(frame = frame)
    
    left_boundary = shape_centre_x - shape_width // 2
    right_boundary = shape_centre_x + shape_width // 2
    top_boundary = shape_centre_y - shape_height // 2
    bottom_boundary = shape_centre_y + shape_height // 2
    
    cv2.rectangle(
        frame,
        (left_boundary, top_boundary),
        (right_boundary, bottom_boundary),
        shape_color,
        cv2.FILLED
    )
    
    if list_landmarks != None and list_landmarks != []:
        cursor = list_landmarks[8]
        
        if left_boundary < cursor[1] < right_boundary and \
            top_boundary < cursor[2] < bottom_boundary:
            shape_color = color_green
            shape_centre_x = cursor[1]
            shape_centre_y = cursor[2]
        else:
            shape_color = color_purple
            
            
    cv2.imshow("Webcam", frame)
    cv2.waitKey(1)
            
  