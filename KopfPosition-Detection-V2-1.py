import cv2
import time
import os
import PoseModulV2 as pm
import math

cap = cv2.VideoCapture(0)
pTime = 0
detector = pm.poseDetector(detectionCon=0.7)
area = 0
while True:
    success, img = cap.read()
    img = detector.findPose(img)
    lmList, bbox = detector.findPosition(img, draw=True)

    if len(lmList) != 0:
        # print(lmList)
        # print(lmList[12lmList[11]],lmList[0])

        # Fiter based on size
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) // 1000
        #print(area)
        if 150 < area < 280:
            #print("yes")
            x1, y1 = lmList[12][1], lmList[12][2]  # Point 2
            x2, y2 = lmList[11][1], lmList[11][2]  # Point 3
            Ax, Ay = (x1 + x2) // 2, (y1 + y2) // 2  # Center - auf X und Y

            length = math.hypot(x2 - x1, y2 - y1)
            # X-Axe
            length1 = Ax+15 #Links
            #print(length1)
            length2 = Ax-25 #Rechts
            # print(length1)
            #length1 = length + (length // 2)  # Links
            # print(length1)
            #length2 = length - (length // 7)  # Rechts
            # print(length1)
            if lmList[0][1] < length2:
                #text = "Looking Right"
                print("Looking Right")

            elif lmList[0][1] > length1:
                #text = "Looking Left"
                print("Looking Left")
            elif length2 < lmList[0][1] < length1:
                #text = "Forward"
                print("Forward")


            # print(lmList[14])
            # cv2.circle(img, (lmList[0][1], lmList[0][2]), 10, (0, 0, 255), cv2.FILLED)



    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (400, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)