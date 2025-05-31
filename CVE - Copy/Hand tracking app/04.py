# Draw 21 landmark on hands, print fps
# Detect index finger tip and draw a circle (use for Finger Tracking, Gesture Recognition, Virtual Drawing)

import cv2 # import OpenCV Package
import mediapipe as mp # import MediaPipe Package
import time # import Time Package

webcam = cv2.VideoCapture(0) # Create webcam object

# Create the Hands object
mp_hands = mp.solutions.hands # Referencing the class
hands = mp_hands.Hands(
    static_image_mode = False,
    max_num_hands = 2,
    min_detection_confidence = 0.5,
    min_tracking_confidence = 0.5
)

mp_draw = mp.solutions.drawing_utils # Referencing the class

time_previous = 0 # Initialise as 0
time_current = 0 # Initialise as 0

while True:
    response, frame = webcam.read() # frame from OpenCV is BGR

    # Converting BGR to RGB format
    frame_RGB = cv2.cvtColor(
        src = frame,
        code = cv2.COLOR_BGR2RGB
    )

    # Hand Tracking Result
    results = hands.process(frame_RGB)

    if results.multi_hand_landmarks != None:
        for each_hand in results.multi_hand_landmarks:
            for each_id, each_landmark in enumerate(each_hand.landmark):
                height, width, channels = frame.shape   # Get size of image
                # Convert normalized x, y coordinates into pixel values
                centre_x = int(each_landmark.x * width) 
                centre_y = int(each_landmark.y * height)

                if each_id == 8: # Tip of Index Finger
                    cv2.circle(
                        img = frame,
                        center = (centre_x, centre_y), # Pixel coordinate
                        radius = 15,                   # How big you want to draw circle
                        color = (255, 0, 255),
                        thickness = cv2.FILLED         
                    )

            mp_draw.draw_landmarks(
                frame,
                each_hand,
                mp_hands.HAND_CONNECTIONS
            )
    
    # Calculate the frames per second
    time_current = time.time()
    fps = 1 / (time_current - time_previous)
    time_previous = time_current

    # Display FPS on Screen
    cv2.putText(
        img = frame,
        text = f'FPS: {str(int(fps))}',
        org = (10, 70),
        fontFace = cv2.FONT_HERSHEY_PLAIN,
        fontScale = 3,
        color = (255, 0, 255),
        thickness = 3
    )

    cv2.imshow("WEBCAM", frame)
    cv2.waitKey(1)