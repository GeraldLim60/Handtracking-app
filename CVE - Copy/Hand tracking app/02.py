import cv2 # import OpenCV Package
import mediapipe as mp # import MediaPipe Package

webcam = cv2.VideoCapture(0) # Create webcam object

# Create the Hands object
mp_hands = mp.solutions.hands # Referencing the class
hands = mp_hands.Hands( #initialize Hands class
    static_image_mode = False, # for real-time video processing
    max_num_hands = 2,
    min_detection_confidence = 0.5, # when the model start scanning input frame to detect the presence of objects 
    min_tracking_confidence = 0.5 # tracking of those landmarks over subsequent frames
)

mp_draw = mp.solutions.drawing_utils # Referencing the class

while True:
    response, frame = webcam.read() # frame from OpenCV is BGR, response in boolean
    
    # Converting BGR to RGB format (Mediapipe require input in RGB)
    frame_RGB = cv2.cvtColor(
        src = frame,
        code = cv2.COLOR_BGR2RGB
    )

    # Hand Tracking Result
    results = hands.process(frame_RGB)

    if results.multi_hand_landmarks != None: #if hands detected
        for each_hand in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                frame,
                each_hand, # 21 landmarks of the detected hand
                mp_hands.HAND_CONNECTIONS
            )
    
    cv2.imshow("WEBCAM", frame)
    cv2.waitKey(1)