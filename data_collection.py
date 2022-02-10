import cv2
import os
# from CV import HandTrackingModuleV2

cap = cv2.VideoCapture(0)
# detector = HandTrackingModuleV2.HandDetector(detectionCon=0.8, maxHands=1)

Gestures = ["0_Palm", "1_L", "2_Fist", "3_One", "4_Five", "5_Straight", "6_ThumbsUp", "7_C", "8_Swing"]

for gest in Gestures:
    if not os.path.exists("data/grayset1/"+gest):
        os.mkdir("data/grayset1/"+gest)


for folder in Gestures:
    #using count variable to name the images in the dataset.
    count = 0

    #Taking input to start the capturing
    print("Press '' to start data collection for "+ folder)
    userinput = input()
    if userinput != '':
        print("Wrong Input..........")
        exit()

    #clicking 500 images per label, you could change as you want.
    while count < 500:

        #read returns two values one is the exit code and other is the frame
        status, img = cap.read()

        #check if we get the frame or not
        if not status:
            print("Frame is not been captured..Exiting...")
            break

        # Find the hand
        # hands, img = detector.findHands(img, draw=False)

        # if hands:
        #     # Hand 1
        #     hand1 = hands[0]
        #     bbox1 = hand1["bbox"]  # Bounding box info x,y,w,h
        #
        # #crop the image
        # crop_img = img[bbox1[1]:bbox1[1] + bbox1[3], bbox1[0]:bbox1[0] + bbox1[2]]

        #convert the image into gray format for fast calculation
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        #display window with gray image
        cv2.imshow("Video Window", gray)

        #resizing the image to store it
        gray = cv2.resize(gray, (500, 500))

        #Store the image to specific label folder
        cv2.imwrite('C:/Users/Hasan/Desktop/SYSC4907/data/grayset1/'+folder+'/img'+str(count)+'.png', gray)

        count = count + 1

        #to quite the display window press 'q'
        if cv2.waitKey(1) == ord('q'):
            break

# When everything done, release the capture
cap.release()