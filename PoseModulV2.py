import cv2
import mediapipe as mp
import time
import math


class poseDetector():
    #########
    #mediapipe.python.solutions.pose.Pose def __init__(self,
             #static_image_mode: bool = False,
             #model_complexity: int = 1,
             #smooth_landmarks: bool = True,
             #enable_segmentation: bool = False,
             #smooth_segmentation: bool = True,
             #min_detection_confidence: float = 0.5,
             #min_tracking_confidence: float = 0.5) -> None

    ########

    def __init__(self, mode=False, ModelCom=1, smoothLand=True,
                 enableSag=False, smoothSag=True, detectionCon=0.5, trackCon=0.5):

        self.mode = mode
        self.ModelCom = ModelCom
        self.smoothLand = smoothLand
        self.smoothSag = smoothSag
        self.enableSag = enableSag
        self.detectionCon = detectionCon
        self.trackCon = trackCon


        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, self.ModelCom, self.smoothLand, self.smoothSag, self.enableSag,
                                     self.detectionCon, self.trackCon)

    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        return img

    def findPosition(self, img, draw=True):
        xList = []
        yList = []
        bbox = []
        self.lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                # print(id, lm)
                cx, cy = int(lm.x * w), int(lm.y * h)
                xList.append(cx)
                yList.append(cy)
                self.lmList.append( [id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)
            bbox = xmin, ymin, xmax, ymax
            if draw:
                cv2.rectangle(img, (bbox[0]-20, bbox[1]-20), (bbox[2]+20, bbox[3]+20), (0, 255, 255), 2)

        return self.lmList, bbox



def main():
    cap = cv2.VideoCapture(0)
    pTime = 0
    detector = poseDetector()
    while True:
        success, img = cap.read()
        img = detector.findPose(img)
        lmList = detector.findPosition(img, draw=False)
        if len(lmList) != 0:
            print(lmList[0])
            cv2.circle(img, (lmList[0][1], lmList[0][2]), 10, (0, 0, 255), cv2.FILLED)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()
