import mediapipe as mp
import numpy as np
#Estas librerias son para crear carpetas de destino y controlar los procesos de 
import os
import multiprocessing
import gc
import cv2
import time

MAX_HISTORIAL = 5
malos = []

#Función que establece el parametro biomecanico del inicio del video
#Recordar que el (0,0) esta en la esquina superior izquierda
def inicio_video(landmarks,mp_pose):
    rodilla = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y 
    talon = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y
    if (rodilla>talon):
        return True
    else: 
        return False

#Función que establece el parametro biomecanico del fin del video
#El bool del inicio debe ser true para que pueda encontrar un fin del video
#El tobillo de la pierna de pateo debe ser mayor que el tobillo de la pierna de no pateo: 
    #en laterales es mayor en la coordenada X
    #en tasera es mayor en la coordenada Z
def fin_video_coordenadas(camara,landmarks,mp_pose, inicio):
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


#Función para verificar que la punta de la pierna derecha esta adelante del talon en la pierna izquierda
def derecha_atras(landmarks,mp_pose, camara): 
    if camara == "T":
        return (landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].z < landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].z)
    elif camara == "LD":
        return (landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].x < landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x)
    elif camara == "LI":
        return (landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].x > landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x)



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
        if cambio_pateo > 0.05 or cambio_no_pateo > 0.05:
            return True #Si cambio la pierna

    return False


#Función que se encarga de cortar los videos dado el frame de inicio y el final. 
def cortar_video(path, path_destino, frame_inicio, frame_final):
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

#Función para evitar varios procesos al tiempo
def cortar_video_en_proceso(path, path_destino, frame_inicio, frame_final):
    proceso = multiprocessing.Process(target=cortar_video, args=(path, path_destino, frame_inicio, frame_final))
    proceso.start()
    proceso.join()  

#Lectura inicial del video. 
def lectura_video(path):
    historial_tobillo_pateo = []
    historial_tobillo_no_pateo = []

    mp_marcar = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose


    captura = cv2.VideoCapture(path,cv2.CAP_FFMPEG)

    if not captura.isOpened():
        print("Error al abrir el video")
        exit()

    ret, frame = captura.read()
    total_frames = int(captura.get(cv2.CAP_PROP_FRAME_COUNT))  
    h, w, _ = frame.shape
    scale_factor = 1
    num_frame_inicio = 0
    num_frame_final = 0
    numero_frame = 0
    bandera_rodilla = False
    bandera_fin = False
    camara = ""
    pausa = False  # Variable para pausar el video

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
                    
                        #if camara == "T":
                        if fin_video_coordenadas(camara,landmarks,mp_pose,bandera_rodilla):
                            bandera_fin = True 
                        # elif camara == "LI" or camara == "LD":
                        #     if fin_video_coordenadas(camara,landmarks,mp_pose,bandera_rodilla):
                        #         bandera_fin = True 
                                
                        # if bandera_rodilla == False:
                        #     mp_marcar.draw_landmarks(
                        #         frame,
                        #         resultados.pose_landmarks,
                        #         mp_pose.POSE_CONNECTIONS,
                        #         mp_marcar.DrawingSpec(color=(0, 0, 255), thickness=1, circle_radius=3),
                        #         mp_marcar.DrawingSpec(color=(255, 255, 255), thickness=1, circle_radius=1)
                        #     )
                        # else:
                        #     if bandera_fin == False: 
                        #         mp_marcar.draw_landmarks(
                        #             frame,
                        #             resultados.pose_landmarks,
                        #             mp_pose.POSE_CONNECTIONS,
                        #             mp_marcar.DrawingSpec(color=(255, 0, 0), thickness=1, circle_radius=3),
                        #             mp_marcar.DrawingSpec(color=(255, 255, 255), thickness=1, circle_radius=1)
                        #         )
                        #     else:
                        #         mp_marcar.draw_landmarks(
                        #             frame,
                        #             resultados.pose_landmarks,
                        #             mp_pose.POSE_CONNECTIONS,
                        #             mp_marcar.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=3),
                        #             mp_marcar.DrawingSpec(color=(255, 255, 255), thickness=1, circle_radius=1)
                        #         )

                    #cv2.imshow("Mediapipe Pose", frame)

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


#Función de lectura_auxiliar del video
def lectura_auxiliar(path):
    pausa = False
    camara = path.split('_')[1]
    captura = cv2.VideoCapture(path, cv2.CAP_FFMPEG)
    total_frames = int(captura.get(cv2.CAP_PROP_FRAME_COUNT))  
    mp_marcar = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    frame_final = int(captura.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_corte_final = None

    if camara == "T":
        mejor_distancia = float('-inf')
    elif camara == "LI":
        mejor_distancia = float('-inf')
    elif camara == "LD":
        mejor_distancia = float('-inf')
    
    num_frame = 0
    index_frame = 0
    with mp_pose.Pose(min_detection_confidence=0.80, min_tracking_confidence=0.90, model_complexity=2) as pose: 
        while captura.isOpened():
            historial_tobillo_pateo = []
            historial_tobillo_no_pateo = []

            if not pausa:  # Solo leer un nuevo frame si no está en pausa
                ret, frame = captura.read()
                if ret:
                    index_frame += 1
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

                        distancia = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y - landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y
                        #print(landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y," - ", landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y)
                        #print(index_frame," -- ",distancia)
                        if (distancia > mejor_distancia and derecha_atras(landmarks,mp_pose, camara)):
                            #print("Mejora")
                            #print(landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].z," - ", landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].z)
                            mejor_distancia = distancia
                            num_frame = index_frame

                        # if camara == "T":
                        #     distancia = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y - landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y
                        #     print(landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y," - ", landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y)
                        #     print(index_frame," -- ",distancia)
                        #     if (distancia > mejor_distancia and derecha_atras(landmarks,mp_pose, camara)):
                        #         print("Mejora")
                        #         print(landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].z," - ", landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].z)
                        #         mejor_distancia = distancia
                        #         num_frame = index_frame
                        # elif camara == "LD":
                        #     distancia = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y - landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y
                        #     print(index_frame," -- ",distancia)
                        #     if (distancia > mejor_distancia and derecha_atras(landmarks,mp_pose,camara)):
                        #         print("Mejora")
                        #         mejor_distancia = distancia
                        #         num_frame = index_frame
                        # elif camara == "LI":
                        #     distancia = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y - landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y
                        #     print(index_frame," -- ",distancia)
                        #     if (distancia > mejor_distancia and derecha_atras(landmarks,mp_pose,camara)):
                        #         print("Mejora")
                        #         mejor_distancia = distancia
                        #         num_frame = index_frame

                    #cv2.imshow("Mediapipe Pose", frame)

            key = cv2.waitKey(25) & 0xFF
            if key == ord('q'): 
                break
            elif index_frame >= total_frames:
                break
            elif key == ord('p'):  # Pausar
                pausa = not pausa

        captura.release()
        cv2.destroyAllWindows()

        #Si se encontro el punto de inicio
        if num_frame != 0:
            #print("Entra al final")
            captura = cv2.VideoCapture(path, cv2.CAP_FFMPEG)
            captura.set(cv2.CAP_PROP_POS_FRAMES, num_frame)
            #print(int(captura.get(cv2.CAP_PROP_POS_FRAMES)))
            #print(num_frame, " - ",frame_final)
            for i in range(num_frame,frame_final + 1):
                #print(i)
                ret, frame = captura.read()
                if ret:
                    frame = cv2.resize(frame, (int(frame.shape[1] * 0.6), int(frame.shape[0] * 0.6)))
                    if "trasera" in path.lower():
                        camara = "T"
                        frame = cv2.rotate(frame, cv2.ROTATE_180)  # Rota la imagen 180°
                    elif "LD" in path:
                        camara = "LD"
                    else:
                        camara = "LI"
                        frame = cv2.rotate(frame, cv2.ROTATE_180)  # Rota la imagen 180°
                    #cv2.imshow("Mediapipe Pose", frame)

                    resultados = pose.process(frame)
                    if resultados.pose_landmarks:
                        landmarks = resultados.pose_landmarks.landmark

                        if camara == "T":
                            if fin_video_coordenadas(camara,landmarks,mp_pose,True):
                                frame_corte_final = i
                                break
                        elif camara == "LI" or camara == "LD":
                            if fin_video_coordenadas(camara,landmarks,mp_pose,True):
                                frame_corte_final = i
                                break
                key = cv2.waitKey(30) & 0xFF
                if key == ord('q'):
                    break

            captura.release()
            cv2.destroyAllWindows()

    return(num_frame, (frame_corte_final), (num_frame!=0), frame_corte_final!=None)


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
videos_faltantes  = [
        #11, 24, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 38, 39, 40, 
        #41, 42, 43, 44, 45, 46, 47, 48, 62, 64, 69, 85, 86, 87, 88, 89, 
        90, 91
]
# for i in videos_faltantes: 
#     video_path_T = f"{rutas_videos[0]}{i}.MOV"
#     video_path_LI = f"{rutas_videos[1]}{i}.MOV"
#     video_path_LD =  f"{rutas_videos[2]}{i}.MOV"

#     lectura_auxiliar(video_path_T)
#     lectura_auxiliar(video_path_LD)
#     lectura_auxiliar(video_path_LI)

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

    # videos_faltantes  = [
    #     11, 24, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 38, 39, 40, 
    #     41, 42, 43, 44, 45, 46, 47, 48, 62, 64, 69, 85, 86, 87, 88, 89, 
    #     90, 91
    # ]
    for i in range(64,100): 
        print(f"------------ VAMOS EN EL VIDEO {i} --------------")
        bool_final_T = False
        bool_final_LI = False
        bool_inicio_T = False
        bool_final_LD = False
        bool_inicio_LI = False
        bool_inicio_LD = False


        video_path_T = f"{rutas_videos[0]}{i}.MOV"
        video_path_LI = f"{rutas_videos[1]}{i}.MOV"
        video_path_LD =  f"{rutas_videos[2]}{i}.MOV"

        inicio_T, final_T, bool_inicio_T, bool_final_T = lectura_video(video_path_T)
        if bool_inicio_T and bool_final_T:
            inicio_LI, final_LI, bool_inicio_LI, bool_final_LI = lectura_video(video_path_LI)
            if bool_inicio_LI and bool_final_LI: 
                inicio_LD, final_LD, bool_inicio_LD, bool_final_LD = lectura_video(video_path_LD)


        if not (bool_inicio_T and bool_final_T and bool_inicio_LI and bool_final_LI and bool_inicio_LD and bool_final_LD):
            print(f"Auxiliar{i}")
            inicio_T, final_T, bool_inicio_T, bool_final_T = lectura_auxiliar(video_path_T)
            inicio_LI, final_LI, bool_inicio_LI, bool_final_LI = lectura_auxiliar(video_path_LI)
            inicio_LD, final_LD, bool_inicio_LD, bool_final_LD = lectura_auxiliar(video_path_LD)


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