import pandas as pd
import cv2
import mediapipe as mp
import numpy as np 

mp_pose = mp.solutions.pose

#Solo tren inferior
puntos_izquierda = [
    mp_pose.PoseLandmark.LEFT_HIP,
    mp_pose.PoseLandmark.LEFT_KNEE,
    mp_pose.PoseLandmark.LEFT_ANKLE,
    mp_pose.PoseLandmark.LEFT_HEEL,
    mp_pose.PoseLandmark.LEFT_FOOT_INDEX
]

#Solo tren inferior
puntos_derecha = [
    mp_pose.PoseLandmark.RIGHT_HIP,
    mp_pose.PoseLandmark.RIGHT_KNEE,
    mp_pose.PoseLandmark.RIGHT_ANKLE,
    mp_pose.PoseLandmark.RIGHT_HEEL,
    mp_pose.PoseLandmark.RIGHT_FOOT_INDEX
]

def extraer_landmarks(path, puntos_interes, camara):
    captura = cv2.VideoCapture(path)
    df = pd.DataFrame(columns=["video", "frame", "lado", "punto", "x", "y", "z"])

    num_video = path.split("_")[-1].replace(".MOV", "")

    if not captura.isOpened():
        return df

    lista = []
    num_frame = 1
    with mp_pose.Pose(min_detection_confidence=0.90, min_tracking_confidence=0.95, model_complexity=2) as pose: 
        while captura.isOpened():
            ret, frame = captura.read()
            if not ret:
                break

            resultados = pose.process(frame)

            if resultados.pose_landmarks:
                for punto in puntos_interes:
                    landmark = resultados.pose_landmarks.landmark[punto]
                    lista.append({
                        "video": num_video,
                        "frame": num_frame,
                        "lado": camara,
                        "punto": punto.name,
                        "x": landmark.x,
                        "y": landmark.y,
                        "z": landmark.z
                    })

            num_frame += 1

        df = pd.DataFrame(lista)
        captura.release()
        return df

video = "G:/Mi unidad/Videos Trabajo de Grado/Lateral Izquierda Limpia/Piso_LI_1.MOV"
df_izquierda = extraer_landmarks(video, puntos_izquierda, "izquierda")
print(df_izquierda)