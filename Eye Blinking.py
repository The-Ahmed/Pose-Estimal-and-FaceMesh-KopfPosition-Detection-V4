import cv2
import mediapipe as mp
import time
import numpy as np
import cvzone

###########Hintergrund###Background############
def drawing_output(image, coordinates_right_eye, coordinates_left_eye, blink_counter):
    aux_image = np.zeros(image.shape, np.uint8)
    contours1 = np.array([coordinates_left_eye])
    contours2 = np.array([coordinates_right_eye])
    cv2.fillPoly(aux_image, pts=[contours1], color=(255, 0, 0))
    cv2.fillPoly(aux_image, pts=[contours2], color=(255, 0, 0))

    cv2.imshow("Aux_image", aux_image)
#############################################
###############Eye Position##################
def eye_aspect_ratio(coordinates):
    d_A = np.linalg.norm(np.array(coordinates[1]) - np.array(coordinates[5]))
    d_B = np.linalg.norm(np.array(coordinates[2]) - np.array(coordinates[4]))
    d_C = np.linalg.norm(np.array(coordinates[0]) - np.array(coordinates[3]))

    return (d_A +d_B) / (2 * d_C)
#################################################
cap = cv2.VideoCapture(0)
pTime = 0

mpDraw = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=1, refine_landmarks=True,
                                  min_detection_confidence=0.5, min_tracking_confidence=0.5)
drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=1)
#Eyes closed
EAR_THRESH = 0.15
counter = 0 # at The counter
NUM_FRAMES = 2
blink_counter = 0
timeStart = time.time()
totalTime = 5

id_left_eye = [33, 160, 158, 133, 153, 144]
id_right_eye = [362, 385, 387, 263, 373, 380]

while True:
    success, image = cap.read()
    imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = faceMesh.process(imgRGB)
    coordinates_left_eye = []
    coordinates_right_eye = []

    #image = cv2.flip(image, 1)
    ih, iw, _ = image.shape
    if results.multi_face_landmarks is not None:
        for faceLms in results.multi_face_landmarks:
            for id in id_left_eye:
                x = int(faceLms.landmark[id].x * iw)
                y = int(faceLms.landmark[id].y * ih)
                coordinates_left_eye.append([x, y])
                cv2.circle(image, (x, y), 2, (0, 255, 255), 1)
                cv2.circle(image, (x, y), 1, (128, 0, 255), 1)
            for id in id_right_eye:
                x = int(faceLms.landmark[id].x * iw)
                y = int(faceLms.landmark[id].y * ih)
                coordinates_right_eye.append([x, y])
                cv2.circle(image, (x, y), 2, (128, 0, 255), 1)
                cv2.circle(image, (x, y), 1, (0, 255, 255), 1)
        ear_left_eye = eye_aspect_ratio(coordinates_left_eye)
        ear_right_eye = eye_aspect_ratio(coordinates_right_eye)
        #print("ear_left_eye:", ear_left_eye, "ear_right_eye", ear_right_eye)
        ear = (ear_left_eye + ear_right_eye)/2
        #print(ear)

        #Eyes Closed #Augen ZU
        if ear < EAR_THRESH:
            counter = 1 #counter Start
            print("Eyes Closed")
            print(counter)

        else:
            if counter >= NUM_FRAMES:
                counter = 0
                blink_counter += 1
                #print(blink_counter)
        drawing_output(image, coordinates_right_eye, coordinates_left_eye, blink_counter) #Hintergrund




        #cTime = time.time()
        #fps = 1 / (cTime - pTime)
        #pTime = cTime
        #cv2.putText(image, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

            # time counter for sliping
            #cvzone.putTextRect(image, 'Time: 20', (450, 25), scale=2, offset=4) #Timer

    cv2.imshow("Eye Tracking", image)
    key = cv2.waitKey(1)


cap.release()
cv2.destroyAllWindows()