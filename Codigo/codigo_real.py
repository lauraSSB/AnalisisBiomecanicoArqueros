import cv2
import mediapipe as mp
import numpy as np

def angulos(punto1, punto2, punto3):
    # El punto 2 es sobre el cual se va a sacar el ángulo
    p1 = np.array(punto1)
    p2 = np.array(punto2)
    p3 = np.array(punto3)

    # Calcular los vectores entre los puntos, respecto al punto 2
    vector1 = p1 - p2
    vector2 = p3 - p2

    # Producto punto entre vectores
    producto_punto = np.dot(vector1, vector2)
    norma_vector1 = np.linalg.norm(vector1)
    norma_vector2 = np.linalg.norm(vector2)

    # Fórmula de ángulo respecto a 2 vectores
    coseno_theta = producto_punto / (norma_vector1 * norma_vector2)
    angulo = np.degrees(np.arccos(coseno_theta))
    return angulo

def coordenadas(rodilla,tobillo):
    if (rodilla<tobillo):
        return True
    else: 
        return False

# Para dibujar la pose en el frame
mp_marcar = mp.solutions.drawing_utils
# Para detectar la pose
mp_pose = mp.solutions.pose

captura = cv2.VideoCapture('pruebaIOS.MOV',cv2.CAP_FFMPEG)

if not captura.isOpened():
    print("Error al abrir el video")
    exit()

ancho = int(captura.get(cv2.CAP_PROP_FRAME_WIDTH))
alto = int(captura.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(captura.get(cv2.CAP_PROP_FPS))
total_frames = int(captura.get(cv2.CAP_PROP_FRAME_COUNT))
ret, frame = captura.read()
h, w, _ = frame.shape
scale_factor = 1

bandera_rodilla = False
pausa = False  # Variable para pausar el video

print("Ancho:", ancho, "Alto:", alto, "FPS:", fps, "Total_frames:", total_frames)

with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.5, model_complexity=2) as pose: 
    while captura.isOpened():
        if not pausa:  # Solo leer un nuevo frame si no está en pausa
            ret, frame = captura.read()

            if ret:
                frame = cv2.resize(frame, (int(frame.shape[1] * scale_factor), int(frame.shape[0] * scale_factor)))
                #frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                resultados = pose.process(frame)

                frame.flags.writeable = True

                if resultados.pose_landmarks:
                    landmarks = resultados.pose_landmarks.landmark

                    if coordenadas(landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y , landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y):
                        bandera_rodilla = True
                    print(bandera_rodilla)

                    if bandera_rodilla == False:
                        mp_marcar.draw_landmarks(
                            frame,
                            resultados.pose_landmarks,
                            mp_pose.POSE_CONNECTIONS,
                            mp_marcar.DrawingSpec(color=(0, 0, 255), thickness=1, circle_radius=1),
                            mp_marcar.DrawingSpec(color=(255, 255, 255), thickness=1, circle_radius=1)
                        )
                    else:
                        mp_marcar.draw_landmarks(
                            frame,
                            resultados.pose_landmarks,
                            mp_pose.POSE_CONNECTIONS,
                            mp_marcar.DrawingSpec(color=(255, 0, 0), thickness=1, circle_radius=3),
                            mp_marcar.DrawingSpec(color=(255, 255, 255), thickness=1, circle_radius=1)
                        )

                    cadera = (landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x * w,
                              landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y * h,
                              landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].z * w)  # Se usa 'w' para escalar z

                    rodilla = (landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x * w,
                               landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y * h,
                               landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].z * w)

                    tobillo = (landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x * w,
                               landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y * h,
                               landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].z * w)

                    # Calcular el ángulo de la rodilla derecha
                    angulo = angulos(cadera, rodilla, tobillo)
                    
                    cv2.putText(frame, f"Tobillo Der: {int(rodilla[1])}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.putText(frame, f"Rodilla Der: {int(tobillo[1])}", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.putText(frame, f"Angulo: {(angulo)}", (70, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        # for x in range(0, w, 10):  # Líneas verticales
        #     cv2.line(frame, (x, 0), (x, h), (200, 200, 200), 1)  # Gris tenue

        # for y in range(0, h, 10):  # Líneas horizontales
        #     cv2.line(frame, (0, y), (w, y), (200, 200, 200), 1)
        cv2.imshow("Mediapipe Pose", frame)

        key = cv2.waitKey(25) & 0xFF
        if key == ord('q'):  # Salir del programa
            break
        elif key == ord('p'):  # Pausar/reanudar
            pausa = not pausa

captura.release()
cv2.destroyAllWindows()
