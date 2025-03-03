import cv2
import mediapipe as mp
import os
import numpy as np

# Función para determinar si ha iniciado el movimiento
def inicio_video(landmarks, mp_pose):
    rodilla = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y
    tobillo = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y
    return rodilla > tobillo

# Función para determinar si ha finalizado el movimiento
def fin_video_coordenadas(camara, landmarks, mp_pose, inicio):
    tobillo_pateo = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x
    rodilla_pateo = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x
    talon_pateo = landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].x
    punta_pateo = landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].x

    tobillo_no_pateo = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x
    rodilla_no_pateo = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x
    talon_no_pateo = landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].x
    punta_no_pateo = landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].x

    if camara == "T" and inicio:
        tobillo_pateo = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].z
        rodilla_pateo = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].z
        tobillo_no_pateo = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].z
        rodilla_no_pateo = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].z
        return tobillo_pateo > tobillo_no_pateo and rodilla_pateo > rodilla_no_pateo

    elif camara in ["LD", "LI"] and inicio:
        return (tobillo_pateo > tobillo_no_pateo and rodilla_pateo > rodilla_no_pateo and
                talon_pateo > talon_no_pateo and punta_pateo > punta_no_pateo)

    return False

# Función para dibujar solo las piernas con colores diferenciados
def dibujar_piernas(frame, landmarks, mp_pose):
    PIERNA_DERECHA = [
        (mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE),
        (mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.RIGHT_ANKLE),
        (mp_pose.PoseLandmark.RIGHT_ANKLE, mp_pose.PoseLandmark.RIGHT_HEEL),
        (mp_pose.PoseLandmark.RIGHT_ANKLE, mp_pose.PoseLandmark.RIGHT_FOOT_INDEX),
        (mp_pose.PoseLandmark.RIGHT_HEEL, mp_pose.PoseLandmark.RIGHT_FOOT_INDEX),
    ]

    PIERNA_IZQUIERDA = [
        (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE),
        (mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.LEFT_ANKLE),
        (mp_pose.PoseLandmark.LEFT_ANKLE, mp_pose.PoseLandmark.LEFT_HEEL),
        (mp_pose.PoseLandmark.LEFT_ANKLE, mp_pose.PoseLandmark.LEFT_FOOT_INDEX),
        (mp_pose.PoseLandmark.LEFT_HEEL, mp_pose.PoseLandmark.LEFT_FOOT_INDEX),
    ]

    h, w, _ = frame.shape

    # Dibujar pierna derecha (Azul)
    for connection in PIERNA_DERECHA:
        p1 = landmarks[connection[0].value]
        p2 = landmarks[connection[1].value]
        cv2.line(frame, (int(p1.x * w), int(p1.y * h)), (int(p2.x * w), int(p2.y * h)), (255, 0, 0), 2)

    # Dibujar pierna izquierda (Verde)
    for connection in PIERNA_IZQUIERDA:
        p1 = landmarks[connection[0].value]
        p2 = landmarks[connection[1].value]
        cv2.line(frame, (int(p1.x * w), int(p1.y * h)), (int(p2.x * w), int(p2.y * h)), (0, 255, 0), 2)

# Función principal para procesar los videos
def lectura_video(path):
    print(path)
    mp_pose = mp.solutions.pose
    captura = cv2.VideoCapture(path, cv2.CAP_FFMPEG)

    if not captura.isOpened():
        print("Error al abrir el video")
        return

    ancho = int(captura.get(cv2.CAP_PROP_FRAME_WIDTH))
    alto = int(captura.get(cv2.CAP_PROP_FRAME_HEIGHT))
    num_frame_inicio = 0

    bandera_rodilla = False
    bandera_fin = False
    camara = ""
    pausa = False
    x = 0
    print("Ancho:", ancho, "Alto:", alto)

    with mp_pose.Pose(min_detection_confidence=0.90, min_tracking_confidence=0.95, model_complexity=2) as pose:
        while captura.isOpened():
            if not pausa:
                ret, frame = captura.read()
                if not ret:
                    break

                x+=1

                if "trasera" in path.lower():
                    camara = "T"
                    frame = cv2.rotate(frame, cv2.ROTATE_180)
                elif "LD" in path:
                    camara = "LD"
                else:
                    camara = "LI"

                resultados = pose.process(frame)
                frame.flags.writeable = True

                if resultados.pose_landmarks:
                    landmarks = resultados.pose_landmarks.landmark

                    if inicio_video(landmarks, mp_pose):
                        bandera_rodilla = True

                    if bandera_rodilla and fin_video_coordenadas(camara, landmarks, mp_pose, bandera_rodilla):
                        bandera_fin = True

                    # Dibujar solo piernas con colores diferenciados
                    dibujar_piernas(frame, landmarks, mp_pose)

                if(bandera_fin == True or x>=189 ):
                        c = "C:/Users/laura/OneDrive/Documents/TrabajoGrado_LauraSalamanca/Frames"
                        os.makedirs(c, exist_ok=True)

                        output_path = os.path.join(c, f"frame_raro_{x}.jpg")
                        cv2.imwrite(output_path, frame)
                        if bandera_fin:
                            exit()
                cv2.imshow("Mediapipe Pose - Piernas Coloreadas", frame)

            key = cv2.waitKey(25) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('p'):
                pausa = not pausa

    captura.release()
    cv2.destroyAllWindows()

# Correr todos los videos
rutas_videos = [
    "C:/Users/laura/OneDrive - Pontificia Universidad Javeriana/Videos Tesis/Saques de Piso/LateralDerecha (Lau)/Piso_LD_",
    "C:/Users/laura/OneDrive - Pontificia Universidad Javeriana/Videos Tesis/Saques de Piso/LateralIzquierda (Sofi)/Piso_LI_",
    "C:/Users/laura/OneDrive - Pontificia Universidad Javeriana/Videos Tesis/Saques de Piso/Trasera(Andy)/Piso_T_"
]

for i in range(5, 38):  # De 4 a 37
    for ruta_base in rutas_videos:
        video_path = f"{ruta_base}{i}.MOV"
        lectura_video(video_path)
