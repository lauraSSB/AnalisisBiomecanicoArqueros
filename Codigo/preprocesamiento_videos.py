import mediapipe as mp
import numpy as np
import os
import gc
import cv2
import time
from ultralytics import YOLO
import multiprocessing

MAX_HISTORIAL = 5
model = YOLO("yolov8n.pt",verbose = False)  # Modelo YOLOv8 nano (rápido y ligero)
malos = []

#Función que establece el parametro biomecanico del inicio del video
#Recordar que el (0,0) esta en la esquina superior izquierda
def inicio_video(landmarks,mp_pose):
    rodilla = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y 
    tobillo = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y
    if (rodilla>tobillo):
        return True
    else: 
        return False


def detectar_balon(imagen):
    resultados = model(imagen, verbose = False)  # Procesar la imagen con YOLO
    for r in resultados:
        for c in r.boxes.cls:  # Verificar clases detectadas
            if model.names[int(c)] == "sports ball":  # Clase "balón"
                return True  # Se encontró un balón
    return False  # No se encontró ningún balón

#Funçión que revisa si la pierna de pateo esta atrás de la pierna de apoyo 
def pierna_pateo_atras(landmarks, mp_pose, balon_bool):
    print(camara)
    tobillo_pateo = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x
    rodilla_pateo = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x
    talon_pateo = landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].x
    punta_pateo = landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].x
        
    tobillo_no_pateo = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x
    rodilla_no_pateo = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x
    talon_no_pateo = landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].x  
    punta_no_pateo = landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].x


    if (camara == "T" and balon_bool):
        tobillo_pateo = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].z
        rodilla_pateo = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].z
        talon_pateo = landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].z
        punta_pateo = landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].z

        tobillo_no_pateo = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].z
        rodilla_no_pateo = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].z
        talon_no_pateo = landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].z
        punta_no_pateo = landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].z

        if(tobillo_pateo < tobillo_no_pateo and rodilla_pateo < rodilla_no_pateo and talon_pateo < talon_no_pateo and punta_pateo < punta_no_pateo):
            return True
        else:
            return False

    elif (camara == "LD" and balon_bool):
        if(tobillo_pateo < tobillo_no_pateo and rodilla_pateo < rodilla_no_pateo and talon_pateo < talon_no_pateo and punta_pateo < punta_no_pateo):
            return True
        else:
            return False
    elif (camara == "LI" and balon_bool):
        if(tobillo_pateo > tobillo_no_pateo and rodilla_pateo > rodilla_no_pateo and talon_pateo > talon_no_pateo and punta_pateo > punta_no_pateo):
            return True
        else:
            return False
    else:
        return False


#def inicio_video_2(landmarks, mp_pose, )

#Función que establece el parametro biomecanico del fin del video
#El bool del inicio debe ser true para que pueda encontrar un fin del video
#El tobillo de la pierna de pateo debe ser mayor que el tobillo de la pierna de no pateo: 
    #en laterales es mayor en la coordenada X
    #en tasera es mayor en la coordenada Z
def fin_video_coordenadas(camara,landmarks,mp_pose, inicio):
    #print(camara," ",inicio)
    #print(camara)
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
        talon_pateo = landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].z
        punta_pateo = landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].z

        tobillo_no_pateo = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].z
        rodilla_no_pateo = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].z
        talon_no_pateo = landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].z
        punta_no_pateo = landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].z

        if(tobillo_pateo > tobillo_no_pateo and rodilla_pateo > rodilla_no_pateo and talon_pateo > talon_no_pateo and punta_pateo > punta_no_pateo):
            return True
        else:
            return False

    elif (camara == "LD" and inicio == True):
        if(tobillo_pateo > tobillo_no_pateo and rodilla_pateo > rodilla_no_pateo and talon_pateo > talon_no_pateo and punta_pateo > punta_no_pateo):
            # print("Tobillos: ", tobillo_pateo," - ",tobillo_no_pateo)
            # print("Rodilla: ", rodilla_pateo," - ",rodilla_no_pateo)
            # print("Talon: ", talon_pateo," - ",talon_no_pateo)
            # print("Punta: ", punta_pateo," - ",punta_no_pateo)
            return True
        else:
            return False
    elif (camara == "LI" and inicio == True):
        if(tobillo_pateo < tobillo_no_pateo and rodilla_pateo < rodilla_no_pateo and talon_pateo < talon_no_pateo and punta_pateo < punta_no_pateo):
            return True
        else:
            return False
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

def cortar_video(path, path_destino, frame_inicio, frame_final):
    print(path," -> ", path_destino, " ---- ", frame_inicio, " - ", frame_final)
    captura = cv2.VideoCapture(path,cv2.CAP_FFMPEG)
    
    if not captura.isOpened():
        return

    fps = int(captura.get(cv2.CAP_PROP_FPS))  
    width = int(captura.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(captura.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = int(captura.get(cv2.CAP_PROP_FOURCC))  
    total_frames = int(captura.get(cv2.CAP_PROP_FRAME_COUNT))  

    writer = cv2.VideoWriter(path_destino, fourcc, fps, (width, height))

    captura.set(cv2.CAP_PROP_POS_FRAMES, frame_inicio)

    for i in range(frame_inicio, frame_final + 1):
        ret, frame = captura.read()
        if not ret:
            break
        if "trasera" in path.lower():
            frame = cv2.rotate(frame, cv2.ROTATE_180)  
        elif "LI" in path:
            frame = cv2.rotate(frame, cv2.ROTATE_180)  
        writer.write(frame)

    captura.release()
    writer.release()
    cv2.waitKey(1)  
    time.sleep(0.5)
    cv2.destroyAllWindows()
    del captura, writer
    gc.collect()

def cortar_video_en_proceso(path, path_destino, frame_inicio, frame_final):
    proceso = multiprocessing.Process(target=cortar_video, args=(path, path_destino, frame_inicio, frame_final))
    proceso.start()
    proceso.join()  


def lectura_video(path):
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
    num_frame_final = 0
    numero_frame = 0
    bandera_rodilla = False
    bandera_fin = False
    camara = ""
    pausa = False  # Variable para pausar el video

    print("Ancho:", ancho, "Alto:", alto, "FPS:", fps, "Total_frames:", total_frames)

    with mp_pose.Pose(min_detection_confidence=0.90, min_tracking_confidence=0.95, model_complexity=2) as pose: 
        while captura.isOpened():
            if not pausa:  # Solo leer un nuevo frame si no está en pausa
                ret, frame = captura.read()
                numero_frame += 1
                if ret:
                    if bandera_rodilla == False:
                        num_frame_inicio += 1
                    if bandera_fin == False:
                        num_frame_final += 1

                    frame = cv2.resize(frame, (int(frame.shape[1] * 0.6), int(frame.shape[0] * 0.6)))
                    if "trasera" in path.lower():
                        camara = "T"
                        frame = cv2.rotate(frame, cv2.ROTATE_180)  # Rota la imagen 180°
                    elif "LD" in path:
                        camara = "LD"
                    else:
                        camara = "LI"
                        frame = cv2.rotate(frame, cv2.ROTATE_180)  # Rota la imagen 180°

                    resultados = pose.process(frame)

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
            elif numero_frame >= total_frames:
                break
            elif bandera_rodilla and bandera_fin and num_frame_inicio < num_frame_final:
                break
            elif key == ord('p'):  # Pausar
                pausa = not pausa

    captura.release()
    cv2.destroyAllWindows()
    return(num_frame_inicio, (num_frame_final), bandera_rodilla, bandera_fin)

if __name__ == '__main__':
    multiprocessing.freeze_support()  
    rutas_videos = [
        "G:/Mi unidad/Videos Trabajo de Grado/Trasera (Andy)/Piso_T_",
        "G:/Mi unidad/Videos Trabajo de Grado/Lateral Izquierda (Sofi)/Piso_LI_",
        "G:/Mi unidad/Videos Trabajo de Grado/Lateral Derecha (Lau)/Piso_LD_"
    ]

    rutas_videos_finales = [
        "G:/Mi unidad/Videos Trabajo de Grado/Trasera Limpia/Piso_T_",
        "G:/Mi unidad/Videos Trabajo de Grado/Lateral Izquierda Limpia/Piso_LI_",
        "G:/Mi unidad/Videos Trabajo de Grado/Lateral Derecha Limpia/Piso_LD_"
    ]
    for i in range(1,100): 
        video_path_T = f"{rutas_videos[0]}{i}.MOV"
        video_path_LI = f"{rutas_videos[1]}{i}.MOV"
        video_path_LD =  f"{rutas_videos[2]}{i}.MOV"

        inicio_T, final_T, bool_inicio_T, bool_final_T = lectura_video(video_path_T)
        inicio_LI, final_LI, bool_inicio_LI, bool_final_LI = lectura_video(video_path_LI)
        inicio_LD, final_LD, bool_inicio_LD, bool_final_LD = lectura_video(video_path_LD)

        if bool_inicio_T and bool_final_T and bool_inicio_LI and bool_final_LI and bool_inicio_LD and bool_final_LD:
            cortar_video_en_proceso(f"{rutas_videos[0]}{i}.MOV", f"{rutas_videos_finales[0]}{i}.MOV",inicio_T, final_T)
            cortar_video_en_proceso(f"{rutas_videos[1]}{i}.MOV", f"{rutas_videos_finales[1]}{i}.MOV",inicio_LI, final_LI)
            cortar_video_en_proceso(f"{rutas_videos[2]}{i}.MOV", f"{rutas_videos_finales[2]}{i}.MOV",inicio_LD, final_LD)
            print(f"Video {i} corregido con exito")
        else: 
            print(f"Video {i} NO SE PUDO CORREGIR")
            malos.append(i)

    for elemento in malos: 
        print(elemento) 

#cortar_video("G:/Mi unidad/Videos Trabajo de Grado/Lateral Derecha (Lau)/Piso_LD_1.MOV","G:/Mi unidad/Videos Trabajo de Grado/Lateral Derecha Limpia/Piso_LD_1.MOV",200,500)
# for i in range(1, 100,5): 
#     for ruta_base in rutas_videos:
#         video_path = f"{ruta_base}{i}.MOV"
#         mp_marcar = mp.solutions.drawing_utils
#         mp_pose = mp.solutions.pose
#         balon_bool = False
        
#         captura = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)  # Se eliminó comilla innecesaria

#         if not captura.isOpened():
#             print(f"Error al abrir el video: {video_path}")
#             continue

#         total_frames = int(captura.get(cv2.CAP_PROP_FRAME_COUNT))
#         with mp_pose.Pose(min_detection_confidence=0.90, min_tracking_confidence=0.95, model_complexity=2) as pose: 
#             for frame_index in range(total_frames - 1, -1, -1):
#                 captura.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
#                 ret, frame = captura.read()

#                 frame = cv2.resize(frame, (int(frame.shape[1] * 0.8), int(frame.shape[0] * 0.8)))
#                 if not ret:
#                     break
                
                
#                 # Definir la cámara según el nombre del archivo
#                 if "trasera" in video_path.lower():
#                     camara = "T"
#                     frame = cv2.rotate(frame, cv2.ROTATE_180)  # Rotar la imagen 180°
#                 elif "LD" in video_path:
#                     camara = "LD"
#                 else:
#                     camara = "LI"

#                 resultados = pose.process(frame)  # Convertir a RGB para Mediapipe 

#                 if resultados.pose_landmarks:
#                     landmarks = resultados.pose_landmarks.landmark
#                     if balon_bool == False: 
#                         balon_bool = detectar_balon(frame)
#                     #print(pierna_pateo_atras(landmarks, mp_pose, balon_bool), " Balon: ", balon_bool)
                    
#                     if(pierna_pateo_atras(landmarks,mp_pose,balon_bool) and balon_bool):
#                         print(frame_index)
#                         break
#                 cv2.imshow("Video en reversa", frame)
#                 if cv2.waitKey(25) & 0xFF == ord('q'):
#                     break

#             for i in range(frame_index, -1, -1):
#                 captura.set(cv2.CAP_PROP_POS_FRAMES, i)
#                 ret, frame = captura.read()

#                 frame = cv2.resize(frame, (int(frame.shape[1] * 0.8), int(frame.shape[0] * 0.8)))
#                 if not ret:
#                     break
                
#                 # Definir la cámara según el nombre del archivo
#                 if "trasera" in video_path.lower():
#                     camara = "T"
#                     frame = cv2.rotate(frame, cv2.ROTATE_180)  # Rotar la imagen 180°
#                 elif "LD" in video_path:
#                     camara = "LD"
#                 else:
#                     camara = "LI"

                
#                 resultados = pose.process(frame)  # Convertir a RGB para Mediapipe

#                 if resultados.pose_landmarks:
#                     landmarks = resultados.pose_landmarks.landmark

#                     mp_marcar.draw_landmarks(
#                         frame, resultados.pose_landmarks, mp_pose.POSE_CONNECTIONS,
#                         mp_marcar.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),  # Puntos clave
#                         mp_marcar.DrawingSpec(color=(255, 0, 0), thickness=2)  # Conexiones
#                     )
                    
#                 cv2.imshow("Video en reversa", frame)
#                 time.sleep(2)
#                 if cv2.waitKey(25) & 0xFF == ord('q'):
#                     break

#         captura.release()

# cv2.destroyAllWindows()