import cv2
import mediapipe as mp
import numpy as np

def angulos(punto1, punto2, punto3):
    p1 = np.array(punto1)
    p2 = np.array(punto2)
    p3 = np.array(punto3)

    vector1 = p1 - p2
    vector2 = p3 - p2

    producto_punto = np.dot(vector1, vector2)
    norma_vector1 = np.linalg.norm(vector1)
    norma_vector2 = np.linalg.norm(vector2)

    coseno_theta = producto_punto / (norma_vector1 * norma_vector2)
    angulo = np.degrees(np.arccos(coseno_theta))
    return angulo

def coordenadas(rodilla, tobillo):
    return rodilla > tobillo

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

captura = cv2.VideoCapture('pruebaIOS.MOV',cv2.CAP_FFMPEG)

if not captura.isOpened():
    print("Error al abrir el video")
    exit()

ret, frame = captura.read()
h, w, _ = frame.shape
scale_factor = 1

bandera_rodilla = False
pausa = False

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=2) as pose:
    while captura.isOpened():
        if not pausa:
            ret, frame = captura.read()

            if ret:
                frame = cv2.resize(frame, (int(frame.shape[1] * scale_factor), int(frame.shape[0] * scale_factor)))
                #frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                resultados = pose.process(frame)

                frame.flags.writeable = True

                if resultados.pose_landmarks:
                    landmarks = resultados.pose_landmarks.landmark

                    rodilla = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]
                    tobillo = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
                    cadera = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]

                    if coordenadas(rodilla.y, tobillo.y):
                        color_puntos = (255, 0, 0)
                        bandera_rodilla = True
                    elif (bandera_rodilla == False):
                        color_puntos = (0, 0, 255)

                    puntos_interes = [mp_pose.PoseLandmark.RIGHT_KNEE.value,
                                      mp_pose.PoseLandmark.RIGHT_ANKLE.value]

                    conexion_interes = [(mp_pose.PoseLandmark.RIGHT_KNEE.value,
                                         mp_pose.PoseLandmark.RIGHT_ANKLE.value)]

                    for idx in puntos_interes:
                        x, y = int(landmarks[idx].x * frame.shape[1]), int(landmarks[idx].y * frame.shape[0])
                        cv2.circle(frame, (x, y), 5, color_puntos, -1)

                    for conexion in conexion_interes:
                        inicio = (int(landmarks[conexion[0]].x * frame.shape[1]), int(landmarks[conexion[0]].y * frame.shape[0]))
                        fin = (int(landmarks[conexion[1]].x * frame.shape[1]), int(landmarks[conexion[1]].y * frame.shape[0]))
                        cv2.line(frame, inicio, fin, (255, 255, 255), 2)

                    angulo = angulos(
                        (cadera.x, cadera.y, cadera.z),
                        (rodilla.x, rodilla.y, rodilla.z),
                        (tobillo.x, tobillo.y, tobillo.z)
                    )

                    cv2.putText(frame, f"Angulo Rodilla: {int(angulo)}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow("Rodilla y Tobillo Derecho", frame)

        key = cv2.waitKey(25) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            pausa = not pausa

captura.release()
cv2.destroyAllWindows()
