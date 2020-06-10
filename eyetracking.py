#Reference = https://www.pyimagesearch.com/2017/04/24/eye-blink-detection-opencv-python-dlib/
# https://towardsdatascience.com/real-time-eye-tracking-using-opencv-and-dlib-b504ca724ac6
# https://github.com/mohitwildbeast/Driver-Drowsiness-Detector
# https://experts.umn.edu/en/publications/eye-tracking-for-detection-of-driver-fatigue

import imutils
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import argparse as arg
import cv2
import dlib
import winsound

print('imutils version: ', imutils.__version__)
print('OpenCV version: ', cv2.__version__)

#creating argument parser for running the program via console
argparser = arg.ArgumentParser()
argparser.add_argument("-p", "--predictor", required = True, help="path of facial landmark predictor")
argparser.add_argument("-v", "--video", type=str, default="", help="path to input video file")
args = vars(argparser.parse_args())

# Computing distance between eyes
from scipy.spatial import distance
def aspect_ratio(eye):
    # vertical
    ya = distance.euclidean(eye[1], eye[5])
    yb = distance.euclidean(eye[2], eye[4])

    #horizontal
    x = distance.euclidean(eye[0], eye[3])

    #aspect ratio
    apr = (ya + yb)/(2*x)

    return apr

#Constants of aspect ratio to indicate blink and consecutive frames
AR_THRESHOLD = 0.20 #0.3 is too racist for me! I hope we could do some scan in the future
AR_CONSECUTIVE_FR = 60

#initialize counters
COUNTER = 0
DROWSY = False

#initialize dlib detector and create facial landmark predictor
print("Loading dlib")
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor(args["predictor"])

#get indexes from facial landmarks

l_eye_start, l_eye_end = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
r_eye_start, r_eye_end = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

import time
#video stream
print("start streaming")
#vs = FileVideoStream(args["video"]).start()
#fileStream = True

vs = VideoStream(src=0).start()
fileStream = False
time.sleep(1.0)

#loop frames
while True:
    #check any more frames left in the buffer
    if fileStream and not vs.more():
        break

    #grab frame and resize, convert to greyscale
    frame = vs.read()
    frame = imutils.resize(frame, width = 400)
    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    #detect faces in frame
    faces = detect(grayscale, 0)

    for face in faces:
        #determine landmarks, converting into array
        shape = face_utils.shape_to_np(predict(grayscale, face))
        #extracting left and right eye
        #left eye landmark scanning
        left_eye = shape[l_eye_start:l_eye_end]
        #right eye landmark scanning
        right_eye = shape[r_eye_start:r_eye_end]
        left_eye_aspect_ratio = aspect_ratio(left_eye)
        right_eye_aspect_ratio = aspect_ratio(right_eye)
        average_aspect_ratio = (left_eye_aspect_ratio + right_eye_aspect_ratio) /2.0
        #computing convex hull
        left_hull = cv2.convexHull(left_eye)
        right_hull = cv2.convexHull(right_eye)
        cv2.drawContours(frame, [left_hull], -1, (255,0,0), 1)
        cv2.drawContours(frame, [right_eye], -1, (255,0,0), 1)

        #check aspect ratio below threshold
        if average_aspect_ratio < AR_THRESHOLD:
            COUNTER += 1

        #else:
            # Drowny condition if more than certain frames
            if COUNTER >= AR_CONSECUTIVE_FR:
                DROWSY = True
                #play beep as alarm
                frequency = 2000  # Set Frequency To 2500 Hertz
                duration = 1000  # Set Duration To 1000 ms == 1 second
                winsound.Beep(frequency, duration)
            #restarting the counter
            
            #else:
            #    DROWSY = False
            #COUNTER = 0


        #put the text
        cv2.putText(frame, "Drowsy: {}".format(str(DROWSY)), (10,30),
        cv2.FONT_HERSHEY_PLAIN, 0.8, (255, 0,0), 2)

        cv2.putText(frame, "Eye Aspect Ratio: {}".format(average_aspect_ratio), (250,30),
        cv2.FONT_HERSHEY_PLAIN, 0.8, (255, 0,0), 2)
        DROWSY = False
    #showing frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()