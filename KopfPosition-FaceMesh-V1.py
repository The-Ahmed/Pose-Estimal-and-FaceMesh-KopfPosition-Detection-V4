import cv2
import mediapipe as mp
import time
import numpy as np

cap = cv2.VideoCapture(0)
pTime = 0

mpDraw = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=1, refine_landmarks=True,
                                  min_detection_confidence=0.5, min_tracking_confidence=0.5)
drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=1)

while True:
    success, image = cap.read()
    imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = faceMesh.process(imgRGB)
    face_3d = []
    face_2d = []
    if results.multi_face_landmarks:
        for faceLms in results.multi_face_landmarks:


            for id,lm in enumerate(faceLms.landmark):
                #print(lm)
                ih, iw, ic = image.shape

                #print(id,x,y)

                if id == 33 or id == 263 or id == 1 or id == 61 or id == 291 or id == 199:
                    if id == 1:
                        nose_2d = (lm.x * iw, lm.y * ih)
                        nose_3d = (lm.x * iw, lm.y * ih, lm.z * 3000)
                    x, y = int(lm.x * iw), int(lm.y * ih)

                # Get the 2D Coordinates
                    face_2d.append([x, y])

                # Get the 3D Coordinates
                    face_3d.append([x, y, lm.z])

            # Convert it to the NumPy array
            face_2d = np.array(face_2d, dtype=np.float64)
            # Convert it to the NumPy array
            face_3d = np.array(face_3d, dtype=np.float64)
            # The camera matrix
            focal_length = 1 * iw
            cam_matrix = np.array([[focal_length, 0, ih / 2], [0, focal_length, iw / 2], [0, 0, 1]])

            # The distortion parameters
            dist_matrix = np.zeros((4, 1), dtype=np.float64)

            # Solve PnP
            success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

            # Get rotational matrix
            rmat, jac = cv2.Rodrigues(rot_vec)

            # Get angles
            angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

            # Get the y rotation degree
            x = angles[0] * 360
            y = angles[1] * 360
            z = angles[2] * 360

            # See where the user's head tilting
            if y < -10:
                text = "Looking Left"
                #print("Looking Left")
            elif y > 10:
                text = "Looking Right"
                #print("Looking Right")
            elif x < -10:
                text = "Looking Down"
                #print("Looking Down")
            elif x > 10:
                text = "Looking Up"
                #print("Looking Up")

            else:
                text = "Forward"
                #print("Forward")

            # Display the nose direction
            nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)

            p1 = (int(nose_2d[0]), int(nose_2d[1]))
            p2 = (int(nose_2d[0] + y * 10), int(nose_2d[1] - x * 10))

            cv2.line(image, p1, p2, (255, 0, 0), 3)

            # Display the nose direction
            nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)

            p1 = (int(nose_2d[0]), int(nose_2d[1]))
            p2 = (int(nose_3d[0] + y * 10), int(nose_3d[1] - x * 10))

            cv2.line(image, p1, p2, (255, 0, 0), 3)
            # Add the text on the image
            cv2.putText(image, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(image, "x: " + str(np.round(x, 2)), (500, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(image, "y: " + str(np.round(y, 2)), (500, 125), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(image, "z: " + str(np.round(z, 2)), (500, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(image, f'FPS: {int(fps)}', (20, 450), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)


        mpDraw.draw_landmarks(image=image, landmark_list=faceLms, connections=mpFaceMesh.FACEMESH_TESSELATION,
                              landmark_drawing_spec=None, connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
        mpDraw.draw_landmarks(image=image, landmark_list=faceLms, connections=mpFaceMesh.FACEMESH_CONTOURS,
                              landmark_drawing_spec=None, connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())
        #mpDraw.draw_landmarks(image=image, landmark_list=faceLms, connections=mpFaceMesh.FACEMESH_IRISES,
                              #landmark_drawing_spec=None, connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style())

        #mpDraw.draw_landmarks(image, faceLms, mpFaceMesh.FACEMESH_TESSELATION, drawSpec, drawSpec)
        #mpDraw.draw_landmarks(image, faceLms, mpFaceMesh.FACEMESH_CONTOURS, drawSpec, drawSpec)

    cv2.imshow("Head Pose Estimation", image)

    cv2.waitKey(1)