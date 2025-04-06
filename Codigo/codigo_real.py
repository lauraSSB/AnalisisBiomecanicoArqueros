import mediapipe as mp
import numpy as np
import time
import cv2

# Inicializar MediaPipe Pose
mp_pose = mp.solutions.pose

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Rutas base de los videos
rutas_videos_finales = [
    "G:/Mi unidad/Videos Trabajo de Grado/Trasera Limpia 2/Piso_T_",
    "G:/Mi unidad/Videos Trabajo de Grado/Lateral Izquierda Limpia 2/Piso_LI_",
    "G:/Mi unidad/Videos Trabajo de Grado/Lateral Derecha Limpia 2/Piso_LD_"
]
#Correccion1
valores = [1, 2, 3, 4, 5, 10, 11, 12, 25, 30, 37, 39, 42, 43, 44, 61, 67, 69, 75, 76, 77, 78, 79, 80, 81]
#Correccion2
valores = [1, 2, 3, 4, 5, 10, 11, 12, 44, 61, 67, 69, 75, 76, 77, 79, 80, 81]

# Recorrer videos
for i in (26,27):
    # if i == 92:
    #     continue
    pose = mp_pose.Pose(min_detection_confidence=0.85, min_tracking_confidence=0.90, model_complexity=2)

    video_path_T = f"{rutas_videos_finales[0]}{i}.mp4"
    video_path_LI = f"{rutas_videos_finales[1]}{i}.mp4"
    video_path_LD = f"{rutas_videos_finales[2]}{i}.mp4"

    captura1 = cv2.VideoCapture(video_path_T, cv2.CAP_FFMPEG)
    captura2 = cv2.VideoCapture(video_path_LI, cv2.CAP_FFMPEG)
    captura3 = cv2.VideoCapture(video_path_LD, cv2.CAP_FFMPEG)

    print(f"VIDEO NUMERO {i}")
    print("Trasera: # Frames: ", int(captura1.get(cv2.CAP_PROP_FRAME_COUNT)), " FPS: ", captura1.get(cv2.CAP_PROP_FPS))
    print("Lateral IZ: # Frames: ", int(captura2.get(cv2.CAP_PROP_FRAME_COUNT)), " FPS: ", captura2.get(cv2.CAP_PROP_FPS))
    print("Lateral DE: # Frames: ", int(captura3.get(cv2.CAP_PROP_FRAME_COUNT)), " FPS: ", captura3.get(cv2.CAP_PROP_FPS))

    cv2.namedWindow("Trasera", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Lateral Izquierda", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Lateral Derecha", cv2.WINDOW_NORMAL)

    cv2.moveWindow("Trasera", 0, 400)
    cv2.moveWindow("Lateral Izquierda", 0, 0)
    cv2.moveWindow("Lateral Derecha", 600, 0)

    while True:
        ret1, frame1 = captura1.read()
        # ret2, frame2 = captura2.read()
        # ret3, frame3 = captura3.read()

        # if not ret1 or not ret2 or not ret3:
        #     break

        # Redimensionar
        frame1 = cv2.resize(frame1, (640, 480))
        # frame2 = cv2.resize(frame2, (640, 480))
        # frame3 = cv2.resize(frame3, (640, 480))

        # Procesar pose
        result1 = pose.process(frame1)
        # result2 = pose.process(frame2)
        # result3 = pose.process(frame3)

        # Dibujar resultados en los frames
        landmark_spec = mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=1, circle_radius=2)
        connection_spec = mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=1, circle_radius=2)

        # Dibujo con configuración personalizada
        if result1.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame1, result1.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=landmark_spec,
                connection_drawing_spec=connection_spec)

        # if result2.pose_landmarks:
        #     mp_drawing.draw_landmarks(
        #         frame2, result2.pose_landmarks, mp_pose.POSE_CONNECTIONS,
        #         landmark_drawing_spec=landmark_spec,
        #         connection_drawing_spec=connection_spec)

        # if result3.pose_landmarks:
        #     mp_drawing.draw_landmarks(
        #         frame3, result3.pose_landmarks, mp_pose.POSE_CONNECTIONS,
        #         landmark_drawing_spec=landmark_spec,
        #         connection_drawing_spec=connection_spec)

        # # Mostrar los frames
        # cv2.imshow("Lateral Izquierda", frame2)
        # cv2.imshow("Lateral Derecha", frame3)
        cv2.imshow("Trasera", frame1)

        time.sleep(2)

        # Reproducir con espera de 30ms
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    
    captura1.release()
    # captura2.release()
    # captura3.release()
    cv2.destroyAllWindows()
