import cv2
import math
from hand_tracking import HandDetector

# Helper function to compute distance between two points
def find_distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    return math.hypot(x2 - x1, y2 - y1)

# Initialize webcam
webcam = cv2.VideoCapture(0)
webcam.set(3, 1280)
webcam.set(4, 720)

# Initialize hand detector
detector = HandDetector(min_detection_confidence=0.8)

# Define colors
colour_purple = (255, 0, 255)
colour_green = (0, 255, 0)
shape_colour = colour_purple

# Define shape properties
shape_centre_x = 100
shape_centre_y = 100
shape_width = 200
shape_height = 200

# Click threshold
click_threshold = 100

while True:
    success, frame = webcam.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    frame = detector.find_hands(frame)
    list_landmarks = detector.find_position(frame)

    # Define shape boundaries
    left = shape_centre_x - shape_width // 2
    right = shape_centre_x + shape_width // 2
    top = shape_centre_y - shape_height // 2
    bottom = shape_centre_y + shape_height // 2

    # Draw the shape
    cv2.rectangle(frame, (left, top), (right, bottom), shape_colour, cv2.FILLED)

    # Check if a hand is detected
    if list_landmarks:
        cursor = list_landmarks[8]  # Index finger tip
        thumb = list_landmarks[4]   # Thumb tip

        point_1 = (cursor[1], cursor[2])
        point_2 = (thumb[1], thumb[2])
        length = find_distance(point_1, point_2)

        if length < click_threshold:
            if left < cursor[1] < right and top < cursor[2] < bottom:
                shape_colour = colour_green
                shape_centre_x = cursor[1]
                shape_centre_y = cursor[2]
        else:
            shape_colour = colour_purple

    cv2.imshow('Webcam', frame)

    # Exit on ESC key
    if cv2.waitKey(1) & 0xFF == 27:
        break

webcam.release()
cv2.destroyAllWindows()