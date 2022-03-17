"""
Computer Vision Integrated with Machine Learning
Author: Hasan Baig
Date: February 15 2022
"""

import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from A_Computer_Vision.EyeTrackingModule import EyeDetector

import statistics
from statistics import mode

# global variables
MODEL_NAME = "mp_hand_gesture"
classNameList = []

def main():
    # initialize Eye Detector class for Eye Detection
    eyeDetector = EyeDetector(maxFaces=1)

    # initialize MediaPipe for Hand Detection
    mpHands = mp.solutions.hands
    hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
    mpDraw = mp.solutions.drawing_utils

    # Load the gesture recognizer model
    model = load_model(MODEL_NAME)
    print(model.summary())

    # Load class names
    f = open('gesture.names', 'r')
    classNames = f.read().split('\n')
    f.close()
    print(classNames)

    # Initialize the webcam
    cap = cv2.VideoCapture(0)

    # counter for classname to be sent via MQTT
    reset_counter = 0

    while True:
        # Read each frame from the webcam
        _, frame = cap.read()
        className = ''
        x, y, c = frame.shape

        # Flip the frame vertically
        frame = cv2.flip(frame, 1)
        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Get hand landmark prediction
        result = hands.process(frameRGB)

        # Find the eyes and its landmarks with draw
        frame, eyes = eyeDetector.findFaceMesh(frame)

        # If eyes are detected
        if eyes:
            # Draw contour around detected eyes
            eyeDetector.drawEyeContour(frame, eyes[0])

            # If landmarks for the hands are detected
            if result.multi_hand_landmarks:
                landmarks = []
                for handslms in result.multi_hand_landmarks:
                    for lm in handslms.landmark:
                        lmx, lmy = int(lm.x * x), int(lm.y * y)
                        landmarks.append([lmx, lmy])

                    # Drawing landmarks on each frame
                    mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)

                    # Predict gesture
                    prediction = model.predict([landmarks])
                    classID = np.argmax(prediction)
                    className = classNames[classID]
                    classNameList.append(className)

                    if reset_counter == 10:
                        print(mode(classNameList))
                        reset_counter = 0

        # show the prediction on the frame
        cv2.putText(frame, className, (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                       1, (0,0,255), 2, cv2.LINE_AA)

        # Show the final output
        cv2.imshow("Webcam", frame)
        if cv2.waitKey(1) == ord('q'):
            break

        reset_counter = reset_counter + 1
        print(reset_counter)

    # release the webcam and destroy all active windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()