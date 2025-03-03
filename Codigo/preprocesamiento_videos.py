import cv2
import mediapipe as mp
import numpy as np
import os

MAX_HISTORIAL = 5

#Función que establece el parametro biomecanico del inicio del video
#Recordar que el (0,0) esta en la esquina superior izquierda
def inicio_video(landmarks,mp_pose):
    rodilla = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y 
    tobillo = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y
    if (rodilla>tobillo):
        return True
    else: 
        return False

#Función que establece el parametro biomecanico del fin del video
#El bool del inicio debe ser true para que pueda encontrar un fin del video
#El tobillo de la pierna de pateo debe ser mayor que el tobillo de la pierna de no pateo: 
    #en laterales es mayor en la coordenada X
    #en tasera es mayor en la coordenada Z
def fin_video_coordenadas(camara,landmarks,mp_pose, inicio):
    #print(camara," ",inicio)
    tobillo_pateo = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x
    rodilla_pateo = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x
    talon_pateo = landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].x
    punta_pateo = landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].x
        
    tobillo_no_pateo = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x
    rodilla_no_pateo = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x
    talon_no_pateo = landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].x  
    punta_no_pateo = landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].x

    if (camara == "T" and inicio == True):
        tobillo_pateo = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].z
        rodilla_pateo = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].z
        cadera_pateo = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].z
        
        tobillo_no_pateo = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].z
        rodilla_no_pateo = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].z
        cadera_no_pateo = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].z

        if(tobillo_pateo > tobillo_no_pateo and rodilla_pateo > rodilla_no_pateo):
            return True
        else:
            return False

    elif (camara == "LD" and inicio == True):
        if(tobillo_pateo > tobillo_no_pateo and rodilla_pateo > rodilla_no_pateo and talon_pateo > talon_no_pateo and punta_pateo > punta_no_pateo):
            print("Tobillos: ", tobillo_pateo," - ",tobillo_no_pateo)
            print("Rodilla: ", rodilla_pateo," - ",rodilla_no_pateo)
            print("Talon: ", talon_pateo," - ",talon_no_pateo)
            print("Punta: ", punta_pateo," - ",punta_no_pateo)
            return True
        else:
            return False
    elif (camara == "LI" and inicio == True):
        if(tobillo_pateo < tobillo_no_pateo and rodilla_pateo < rodilla_no_pateo and talon_pateo < talon_no_pateo and punta_pateo < punta_no_pateo):
            return True
        else:
            return Falseaw3i
    else:
        return False

#Función para verificar que los cambios entre coordenadas entre el frame anterior y el actual no sea mayor al 20%
#Se va a hacer todo respecto a los tobillos, que son los principales actores en este caso
def verificar_cambios(landmarks, mp_pose,historial_tobillo_pateo,historial_tobillo_no_pateo):
    tobillo_pateo = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x
    tobillo_no_pateo = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x

    historial_tobillo_pateo.append(tobillo_pateo)
    historial_tobillo_no_pateo.append(tobillo_no_pateo)

    #Verificar que solo hay 5 elementos en el historial
    if len(historial_tobillo_pateo) > MAX_HISTORIAL:
        historial_tobillo_pateo.pop(0)
        historial_tobillo_no_pateo.pop(0)

    if len(historial_tobillo_pateo) >= 2:
        cambio_pateo = abs(historial_tobillo_pateo[-1] - historial_tobillo_pateo[-2])
        cambio_no_pateo = abs(historial_tobillo_no_pateo[-1] - historial_tobillo_no_pateo[-2])
        #print(cambio_pateo," ------ ",cambio_no_pateo)
        if cambio_pateo > 0.05 or cambio_no_pateo > 0.05:
            return True #Si cambio la pierna

    return False

def lectura_video(path):

    #Inicialización de colas para supervisar que no hayan cambios grandes al momento de hacer la lectura
    historial_tobillo_pateo = []
    historial_tobillo_no_pateo = []

    mp_marcar = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    captura = cv2.VideoCapture(path,cv2.CAP_FFMPEG)

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
    num_frame_inicio = 0

    bandera_rodilla = False
    bandera_fin = False
    camara = ""
    pausa = False  # Variable para pausar el video

    print("Ancho:", ancho, "Alto:", alto, "FPS:", fps, "Total_frames:", total_frames)

    with mp_pose.Pose(min_detection_confidence=0.90, min_tracking_confidence=0.95, model_complexity=2) as pose: 
        while captura.isOpened():
            if not pausa:  # Solo leer un nuevo frame si no está en pausa
                ret, frame = captura.read()

                if ret:
                    resultados = pose.process(frame)
                    if "trasera" in path.lower():
                        camara = "T"
                        frame = cv2.rotate(frame, cv2.ROTATE_180)  # Rota la imagen 180°
                    elif "LD" in path:
                        camara = "LD"
                    else:
                        camara = "LI"

                    if resultados.pose_landmarks:
                        landmarks = resultados.pose_landmarks.landmark

                        if verificar_cambios(landmarks, mp_pose, historial_tobillo_pateo, historial_tobillo_no_pateo):
                            print("Recalculando")
                            continue

                        if inicio_video(landmarks,mp_pose):
                            bandera_rodilla = True
                        
                        if camara == "T":
                            if fin_video_coordenadas(camara,landmarks,mp_pose,bandera_rodilla):
                                bandera_fin = True 
                        elif camara == "LI" or camara == "LD":
                            if fin_video_coordenadas(camara,landmarks,mp_pose,bandera_rodilla):
                                bandera_fin = True 
                                
                        if bandera_rodilla == False:
                            num_frame_inicio += 1
                            mp_marcar.draw_landmarks(
                                frame,
                                resultados.pose_landmarks,
                                mp_pose.POSE_CONNECTIONS,
                                mp_marcar.DrawingSpec(color=(0, 0, 255), thickness=1, circle_radius=3),
                                mp_marcar.DrawingSpec(color=(255, 255, 255), thickness=1, circle_radius=1)
                            )
                        else:
                            if bandera_fin == False: 
                                mp_marcar.draw_landmarks(
                                    frame,
                                    resultados.pose_landmarks,
                                    mp_pose.POSE_CONNECTIONS,
                                    mp_marcar.DrawingSpec(color=(255, 0, 0), thickness=1, circle_radius=3),
                                    mp_marcar.DrawingSpec(color=(255, 255, 255), thickness=1, circle_radius=1)
                                )
                            else:
                                mp_marcar.draw_landmarks(
                                    frame,
                                    resultados.pose_landmarks,
                                    mp_pose.POSE_CONNECTIONS,
                                    mp_marcar.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=3),
                                    mp_marcar.DrawingSpec(color=(255, 255, 255), thickness=1, circle_radius=1)
                                )

                    cv2.imshow("Mediapipe Pose", frame)

            key = cv2.waitKey(25) & 0xFF
            if key == ord('q'): 
                break
            elif key == ord('p'):  # Pausar
                pausa = not pausa

    captura.release()
    cv2.destroyAllWindows()
    return(path," - ",num_frame_inicio)

rutas_videos = [
    "C:/Users/laura/OneDrive - Pontificia Universidad Javeriana/Videos Tesis/Saques de Piso/LateralDerecha (Lau)/Piso_LD_"
    #"C:/Users/laura/OneDrive - Pontificia Universidad Javeriana/Videos Tesis/Saques de Piso/LateralIzquierda (Sofi)/Piso_LI_",
    #"C:/Users/laura/OneDrive - Pontificia Universidad Javeriana/Videos Tesis/Saques de Piso/Trasera(Andy)/Piso_T_"
]

for i in range(5,9):  # 38 porque el range() excluye el último número
    for ruta_base in rutas_videos:
        video_path = f"{ruta_base}{i}.MOV"
        print(lectura_video(video_path))
