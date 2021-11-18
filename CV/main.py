# # This is a sample Python script.
#
# # Press Shift+F10 to execute it or replace it with your code.
# # Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
#
#
# def print_hi(name):
#     # Use a breakpoint in the code line below to debug your script.
#     print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.
#
#
# # Press the green button in the gutter to run the script.
# if __name__ == '__main__':
#     print_hi('PyCharm')
#
# # See PyCharm help at https://www.jetbrains.com/help/pycharm/

import cv2
from cvzone.HandTrackingModule import HandDetector

cap = cv2.VideoCapture(0)
detector = HandDetector(detectionCon=0.8, maxHands=2)

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)

    if hands:
        # Hand 1
        hand1 = hands[0]
        lmList1 = hand1["lmList"] # List of 21 landmarks
        bbox1 = hand1["bbox"] # Bounding Box info x,y,w,h
        centerPoint1 = hand1["center"] # Center of Hand cx,cy
        handType1 = hand1["type"] # Hand Type Left/Right

        if len(hands)==2:
            hand2 = hands[1]
            lmList2 = hand2["lmList"]  # List of 21 landmarks
            bbox2 = hand2["bbox"]  # Bounding Box info x,y,w,h
            centerPoint2 = hand2["center"]  # Center of Hand cx,cy
            handType2 = hand2["type"]  # Hand Type Left/Right

            print(handType1,handType2)

    cv2.imshow("Webcam", img)
    cv2.waitKey(1)
