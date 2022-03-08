# organize imports
import cv2
import imutils
import numpy as np
from tensorflow.keras.models import load_model
import paho.mqtt.client as mqtt
from CV.HandTrackingModule import HandDetector
# import ssl
#
# ssl_ctx = ssl.create_default_context()
# ssl_ctx.check_hostname = False
# ssl_ctx.verify_mode = ssl.CERT_NONE
#
# try:
#     _create_unverified_https_context = ssl._create_unverified_context
# except AttributeError:
#     # Legacy Python that doesn't verify HTTPS certificates by default
#     pass
# else:
#     # Handle target environment that doesn't support HTTPS verification
#     ssl._create_default_https_context = _create_unverified_https_context

from CV.EyeTrackingModule import EyeDetector


# global variables
bg = None
IMG_SIZE_X = 120
IMG_SIZE_Y = 320
MODEL_NAME = "my_model_version_2.h5" #"my_model.h5"

# MQTT global
client = ""
payload = "2"
mqtt_clientId = ""
mqtt_username = "testing"
mqtt_password = "Abc12345"
mqtt_host = "f51bc650a9c24db18f2b2d13134a6da1.s1.eu.hivemq.cloud"
mqtt_port = 8883
mqtt_topic_publish = "gestures/gesture1"

def run_avg(image, accumWeight):
    global bg
    # initialize the background
    if bg is None:
        bg = image.copy().astype("float")
        return

    # compute weighted average, accumulate it and update the background
    cv2.accumulateWeighted(image, bg, accumWeight)


def segment(image, threshold=15):
    global bg
    # find the absolute difference between background and current frame
    diff = cv2.absdiff(bg.astype("uint8"), image)

    # threshold the diff image so that we get the foreground
    thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)[1]

    # get the contours in the thresholded image
    (cnts, _) = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # return None, if no contours detected
    if len(cnts) == 0:
        return
    else:
        # based on contour area, get the maximum contour which is the hand
        segmented = max(cnts, key=cv2.contourArea)
        return (thresholded, segmented)


def _load_weights():
    try:
        model = load_model(MODEL_NAME)
        print(model.summary())
        # print(model.get_weights())
        # print(model.optimizer)
        return model
    except Exception as e:
        return None


def getPredictedClass(model):

    image = cv2.imread('Temp.png')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_image = cv2.resize(gray_image, [IMG_SIZE_Y, IMG_SIZE_X])
    gray_image = gray_image.reshape(1, IMG_SIZE_X, IMG_SIZE_Y, 1)

    prediction = model.predict_on_batch(gray_image)
    # print(prediction)
    predicted_class = np.argmax(prediction)

    if predicted_class == 1:
        gesture = "Palm (Horizontal)"
    # elif predicted_class == 0:
    #     gesture = "Thumb down"
    elif predicted_class == 2:
        gesture = "L"
    elif predicted_class == 3:
        gesture = "Fist (Horizontal)"
    elif predicted_class == 4:
        gesture = "Fist (Vertical)"
    # elif predicted_class == 5:
    #     gesture = "Thumbs up"
    elif predicted_class == 6:
        gesture = "Index"
    # elif predicted_class == 7:
    #     gesture = "OK"
    elif predicted_class == 8:
        gesture = "Palm (Vertical)"
    elif predicted_class == 9:
        gesture = "C"
    else:
        gesture = "None"

    # print(gesture)
    return gesture


################

def init_mqtt():
    global client #, mqtt_clientId, mqtt_username, mqtt_password, mqtt_host
    # Set up the client
    client = mqtt.Client(client_id=mqtt_clientId)
    client.username_pw_set(
        username=mqtt_username,
        password=mqtt_password
    )
    # client.tls_set(tls_version=mqtt.ssl.PROTOCOL_TLS)
    # client.tls_insecure_set(True)
    client.connect(mqtt_host, 8883, 60) # certificate error
    # client.connect(mqtt_host) timeout

    # Set up the callbacks
    client.on_publish = on_publish
    client.loop_start()

def on_publish(client: mqtt.Client, userdata, mid, properties=None):
    print("mid: " + str(mid))

# def publish_mqtt_message():
#     global client, mqtt_topic_publish, payload
#     message_info = client.publish(
#         topic=mqtt_topic_publish,
#         payload=payload,
#         qos=0
#     )
#
#     message_info.wait_for_publish()

################

if __name__ == "__main__":
    ################

    # initialize mqtt
    init_mqtt()

    ################

    # initialize accumulated weight
    accumWeight = 0.5

    # get the reference to the webcam
    camera = cv2.VideoCapture(0)

    # handDetector = HandDetector(detectionCon=0.8, maxHands=1)
    eyeDetector = EyeDetector(maxFaces=1)

    fps = int(camera.get(cv2.CAP_PROP_FPS))
    # region of interest (ROI) coordinates
    top, right, bottom, left = 10, 250, 225, 490
    # initialize num of frames
    num_frames = 0
    # calibration indicator
    calibrated = False
    model = _load_weights()
    k = 0
    # keep looping, until interrupted
    while (True):
        # Get image frame
        success, img = camera.read()

        # # Find the hand and its landmarks with draw
        # hands, detection = handDetector.findHands(img)
        # Find the face and its landmarks with draw
        detection, faces = eyeDetector.findFaceMesh(img)

        # get the current frame
        (grabbed, frame) = camera.read()

        # resize the frame
        frame = imutils.resize(frame, width=700)
        # flip the frame so that it is not the mirror view
        frame = cv2.flip(frame, 1)

        # clone the frame
        clone = frame.copy()

        # get the height and width of the frame
        (height, width) = frame.shape[:2]

        # if hands:
        #     bbox = hands[0]["bbox"]
        #     bb_left = bbox[0] - 20
        #     bb_top = bbox[1] - 20
        #     bb_right = bbox[0] + bbox[2] + 20
        #     bb_bottom = bbox[1] + bbox[3] + 20
        #     roi2 = frame[bb_top:bb_bottom, bb_right:bb_left]

        # get the ROI
        roi = frame[top:bottom, right:left]

        # convert the roi to grayscale and blur it
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        # to get the background, keep looking till a threshold is reached
        # so that our weighted average model gets calibrated
        if num_frames < 30:
            run_avg(gray, accumWeight)
            if num_frames == 1:
                print("[STATUS] please wait! calibrating...")
            elif num_frames == 29:
                print("[STATUS] calibration successfull...")
        else:
            # segment the hand region
            hand = segment(gray)

            # check whether hand region is segmented
            if hand is not None and faces:
                face = faces[0]
                eyeDetector.drawEyeContour(detection, face)

                # if yes, unpack the thresholded image and
                # segmented region
                (thresholded, segmented) = hand

                # draw the segmented region and display the frame
                cv2.drawContours(clone, [segmented + (right, top)], -1, (0, 0, 255))

                # count the number of fingers
                # fingers = count(thresholded, segmented)
                if k % (fps / 3) == 0:
                    cv2.imwrite('Temp.png', thresholded)
                    predictedClass = getPredictedClass(model)
                    cv2.putText(clone, str(predictedClass), (70, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    print(predictedClass)
                    ################

                    client.publish(
                        topic=mqtt_topic_publish,
                        payload=payload,
                        qos=2
                    )
                    # print(payload)
                    # publish_mqtt_message()
                    # Thread(target=publish_mqtt_message).start()
                    ################

                # show the thresholded image
                cv2.imshow("Thesholded", thresholded)
        k = k + 1

        # draw the segmented hand
        cv2.rectangle(clone, (left, top), (right, bottom), (0, 255, 0), 2)

        # increment the number of frames
        num_frames += 1

        # display the frame with segmented hand
        cv2.imshow("Video Feed", clone)

        # Display
        detection = cv2.flip(detection, 1)
        cv2.imshow("Detection", detection)
        cv2.waitKey(1)

        # observe the keypress by the user
        keypress = cv2.waitKey(1) & 0xFF

        # if the user pressed "q", then stop looping
        if keypress == ord("q"):
            break

    # free up memory
    camera.release()
    cv2.destroyAllWindows()