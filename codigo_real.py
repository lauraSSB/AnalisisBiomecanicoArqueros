import cv2
import mediapipe as mp
import numpy as np

def angulos(punto1,punto2,punto3):
    #El punto 2 es sobre el cual se va a sacar el angulo
    
    p1 = np.array(punto1)
    p2 = np.array(punto2)
    p3 = np.array(punto3)

    #Calcular los vectores entre los puntos, respecto al punto 2
    vector1 = p1 - p2
    vector2 = p3 - p2

    #Producto punto entre vectores
    producto_punto = np.dot(vector1,vector2)
    norma_vector1 = np.linalg.norm(vector1)
    norma_vector2 = np.linalg.norm(vector2)

    #Formula de angulo respecto a 2 vectores
    coseno_theta = producto_punto/(norma_vector1*norma_vector2)

    angulo = np.degrees(np.arccos(coseno_theta))
    return angulo



#Para dibujar la pose en el frame
mp_marcar = mp.solutions.drawing_utils
#Para detectar la pose
mp_pose = mp.solutions.pose

captura = cv2.VideoCapture('videoprueba2.mp4')

if not captura.isOpened():
    print("Error al abrir el video")

ancho = int(captura.get(cv2.CAP_PROP_FRAME_WIDTH))
alto = int(captura.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(captura.get(cv2.CAP_PROP_FPS))
total_frames = int(captura.get(cv2.CAP_PROP_FRAME_COUNT))

scale_factor = 0.2


print("Ancho: ",ancho," Alto: ", alto," FPS: ", fps, " Total_frames: ",total_frames)

with mp_pose.Pose(min_detection_confidence = 0.5, min_tracking_confidence = 0.5, model_complexity = 2) as pose: 
    while captura.isOpened():
        # ret indica si se leyó correctamente el frame
        # frame es el frame de la captura
        ret, frame = captura.read()

        frame = cv2.resize(frame, (int(frame.shape[1] * scale_factor), int(frame.shape[0] * scale_factor)))

        if ret == True:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resultados = pose.process(frame)

            frame.flags.writeable = True

            if resultados.pose_landmarks:
                mp_marcar.draw_landmarks(
                    frame,
                    resultados.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_marcar.DrawingSpec(color=(0, 0, 255), thickness=1, circle_radius=1),
                    mp_marcar.DrawingSpec(color=(255, 255, 255), thickness=1, circle_radius=1)
                )
                landmarks = resultados.pose_landmarks.landmark
                h,      w, _ = frame.shape  # Obtener dimensiones
                cadera = (landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x * w,
                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y * h,
                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].z * w)  # Se usa 'w' para escalar z

                rodilla = (landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x * w,
                   landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y * h,
                   landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].z * w)

                tobillo = (landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x * w,
                   landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y * h,
                   landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].z * w)

                # Calcular el ángulo de la rodilla derecha
                angulo = angulos(cadera, rodilla, tobillo)
                cv2.putText(frame, f"{int(angulo)}", (int(rodilla[0]) - 30, int(rodilla[1]) - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow("Mediapipe Pose", frame)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
            # end if
        # end if

        else: 
            print("Fin del video o error al leer el frame")
            break
        # end else
    # end while

captura.release()
cv2.destroyAllWindows()
