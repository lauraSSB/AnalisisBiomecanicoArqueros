from django.shortcuts import render, HttpResponse
from django.core.files.storage import FileSystemStorage
import datetime
import cv2
import os
import mediapipe as mp
import numpy as np
from copy import deepcopy
import pandas as pd
import time
import math
from collections import deque

MAX_HISTORIAL = 5
malos = []

def calcular_angulo(p1, p2, p3):
    a = np.array(p1) - np.array(p2)
    b = np.array(p3) - np.array(p2)
    angulo_rad = math.acos(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    return round(np.degrees(angulo_rad),3)

def calcular_angulo_vertical(hombro, cadera):
    v = np.array([hombro[0] - cadera[0], hombro[1] - cadera[1]])
    vertical = np.array([0, -1])

    cos_theta = np.dot(v, vertical) / (np.linalg.norm(v) * np.linalg.norm(vertical))
    angulo = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))

    signo = np.sign(hombro[0] - cadera[0])

    return round(signo * angulo,3)

def calcular_angulo_inclinacion_tronco(medio_hombro, media_cadera, talon):
    talon = np.array([talon[0], talon[1]])

    tronco = np.array(medio_hombro) - np.array(media_cadera)

    eje_vertical = np.array(media_cadera) - talon

    cos_theta = np.dot(tronco, eje_vertical) / (np.linalg.norm(tronco) * np.linalg.norm(eje_vertical))
    angulo = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))

    signo = np.sign(medio_hombro[0] - media_cadera[0])

    return round(signo * angulo,3)

def visibilidad(landmarks, mp_pose):
    visibilidad = [
        landmarks[mp_pose.PoseLandmark.NOSE].visibility,
        landmarks[mp_pose.PoseLandmark.LEFT_HEEL].visibility,
        landmarks[mp_pose.PoseLandmark.RIGHT_HEEL].visibility
    ]

    if all(v > 0.7 for v in visibilidad):
        return True
    else:
        return False

def inicio_video(landmarks,mp_pose):
    rodilla = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y 
    talon = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y
    if (rodilla>talon):
        return True
    else: 
        return False

def fin_video_coordenadas(camara,landmarks,mp_pose, inicio):
    tobillo_pateo = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x
    rodilla_pateo = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x
    talon_pateo = landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].x
    punta_pateo = landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].x
        
    tobillo_no_pateo = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x
    rodilla_no_pateo = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x
    talon_no_pateo = landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].x  
    punta_no_pateo = landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].x

    if (camara == "trasera" and inicio == True):
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

    elif (camara == "derecha" and inicio == True):
        if(tobillo_pateo > tobillo_no_pateo and rodilla_pateo > rodilla_no_pateo and talon_pateo > talon_no_pateo and punta_pateo > punta_no_pateo):
            return True
        else:
            return False
    elif (camara == "izquierda" and inicio == True):
        if(tobillo_pateo < tobillo_no_pateo and rodilla_pateo < rodilla_no_pateo and talon_pateo < talon_no_pateo and punta_pateo < punta_no_pateo):
            return True
        else:
            return False
    else:
        return False

def derecha_atras(landmarks,mp_pose, camara): 
    if camara == "trasera":
        return (landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].z < landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].z 
                and landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].z < landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].z
                and landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y > landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y)
    elif camara == "derecha":
        return (landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].x < landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x)
    elif camara == "izquierda":
        return (landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].x > landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x)

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
        if cambio_pateo > 0.06 or cambio_no_pateo > 0.06:
            return True #Si cambio la pierna

    return False

def corregir_primer_frame(df_distancias, registros, puntos_a_usar, umbral=15):
    registros_corregidos = deepcopy(registros)
    grupo_pierna = list(puntos_a_usar.keys())

    for marcador in grupo_pierna:
        grupo = df_distancias[df_distancias['marcador'] == marcador].sort_values('frame_anterior')

        if len(grupo) >= 3:
            d01 = grupo.iloc[0]['distancia']
            d12 = grupo.iloc[1]['distancia']
            d23 = grupo.iloc[2]['distancia']

            if d01 is not None and d12 is not None and d23 is not None:
                if d01 > umbral and d12 < umbral and d23 < umbral:
                    print("Corrige primer frame")
                    for m in grupo_pierna:
                        frame1 = next((r for r in registros_corregidos if r['marcador'] == m and r['frame'] == 1), None)
                        frame0 = next((r for r in registros_corregidos if r['marcador'] == m and r['frame'] == 0), None)
                        if frame0 and frame1:
                            frame0['X (px)'] = frame1['X (px)']
                            frame0['Y (px)'] = frame1['Y (px)']
                    break

    return registros_corregidos

def corregir_frames_estaticos(df_distancias, registros_video, puntos_a_usar):
    registros_corregidos = deepcopy(registros_video)
    grupo_pierna = list(puntos_a_usar.keys())

    for frame in df_distancias['frame_actual'].unique():
        grupo_frame = df_distancias[df_distancias['frame_actual'] == frame]

        if set(grupo_frame['marcador']) == set(grupo_pierna) and all(grupo_frame['distancia'] == 0):
            print(f"Corrigiendo frame {frame} (todas las distancias 0)")

            for marcador in grupo_pierna:
                puntos_anteriores = []
                puntos_posteriores = []

                for offset in [1, 2]:
                    anterior = next((r for r in registros_corregidos if r['marcador'] == marcador and r['frame'] == frame - offset), None)
                    posterior = next((r for r in registros_corregidos if r['marcador'] == marcador and r['frame'] == frame + offset), None)

                    if anterior and anterior['X (px)'] is not None and anterior['Y (px)'] is not None:
                        puntos_anteriores.append((anterior['X (px)'], anterior['Y (px)']))
                    if posterior and posterior['X (px)'] is not None and posterior['Y (px)'] is not None:
                        puntos_posteriores.append((posterior['X (px)'], posterior['Y (px)']))

                todos_puntos = puntos_anteriores + puntos_posteriores
                if len(todos_puntos) >= 2:
                    xs, ys = zip(*todos_puntos)
                    promedio_x = sum(xs) / len(xs)
                    promedio_y = sum(ys) / len(ys)

                    punto_actual = next((r for r in registros_corregidos if r['marcador'] == marcador and r['frame'] == frame), None)
                    if punto_actual:
                        punto_actual['X (px)'] = promedio_x
                        punto_actual['Y (px)'] = promedio_y

    return registros_corregidos

def corregir_frames_grandes(df_distancias, registros_video, puntos_a_usar, lado):
    registros_corregidos = deepcopy(registros_video)
    df = pd.DataFrame(registros_corregidos)

    if lado == "trasera":
        umbrales_por_marcador = {
            'hombro_derecho': 13,
            'cadera_izquierda': 10,
            'talon_izquierdo': 10,
            'hombro_izquierdo': 13,
            'cadera_derecha': 10,
            'rodilla_izquierda': 18,
            'tobillo_izquierdo': 10
        }
    else:
        umbrales_por_marcador = {
            'cadera_derecha': 22,
            'rodilla_derecha': 60,
            'tobillo_derecho': 130,
            'talon_derecho': 150,
            'punta_derecha': 150,
            'hombro_derecho': 18,
            'cadera_izquierda': 30,
            'rodilla_izquierda': 25,
            'tobillo_izquierdo': 15,
            'talon_izquierdo': 15,
            'punta_izquierda': 15,
            'hombro_izquierdo': 30,
        }

    for _, fila in df_distancias.iterrows():
        marcador = fila['marcador']
        frame_actual = fila['frame_actual']
        frame_anterior = fila['frame_anterior']
        frame_siguiente = frame_actual + 1
        distancia = fila['distancia']

        umbral = umbrales_por_marcador.get(marcador, 20)
        if distancia > umbral:
            punto_anterior = df[(df['frame'] == frame_anterior) & (df['marcador'] == marcador)]
            punto_siguiente = df[(df['frame'] == frame_siguiente) & (df['marcador'] == marcador)]
            punto_actual_idx = df[(df['frame'] == frame_actual) & (df['marcador'] == marcador)].index

            if not punto_actual_idx.empty and not punto_anterior.empty:
                x_anterior = punto_anterior.iloc[0]['X (px)']
                y_anterior = punto_anterior.iloc[0]['Y (px)']

                if not punto_siguiente.empty:
                    x_siguiente = punto_siguiente.iloc[0]['X (px)']
                    y_siguiente = punto_siguiente.iloc[0]['Y (px)']

                    if all(pd.notnull([x_anterior, y_anterior, x_siguiente, y_siguiente])):
                        promedio_x = (x_anterior + x_siguiente) / 2
                        promedio_y = (y_anterior + y_siguiente) / 2
                        df.loc[punto_actual_idx, 'X (px)'] = promedio_x
                        df.loc[punto_actual_idx, 'Y (px)'] = promedio_y
                else:
                    if pd.notnull(x_anterior) and pd.notnull(y_anterior):
                        df.loc[punto_actual_idx, 'X (px)'] = x_anterior
                        df.loc[punto_actual_idx, 'Y (px)'] = y_anterior

    return df.to_dict('records')

def data_frame_distancias(registros):
    df = pd.DataFrame(registros)
    distancias = []

    for marcador, grupo in df.groupby('marcador'):
        grupo = grupo.sort_values('frame')

        for i in range(1, len(grupo)):
            fila_anterior = grupo.iloc[i - 1]
            fila_actual = grupo.iloc[i]

            x1, y1 = fila_anterior['X (px)'], fila_anterior['Y (px)']
            x2, y2 = fila_actual['X (px)'], fila_actual['Y (px)']

            if pd.notnull(x1) and pd.notnull(y1) and pd.notnull(x2) and pd.notnull(y2):
                dx = x2 - x1
                dy = y2 - y1
                distancia = np.sqrt(dx**2 + dy**2)
            else:
                distancia = None 

            distancias.append({
                'marcador': marcador,
                'frame_anterior': fila_anterior['frame'],
                'frame_actual': fila_actual['frame'],
                'distancia': distancia
            })

    return pd.DataFrame(distancias)


def calcular_puntos_medios(registros, puntos_a_usar):
    registros_actualizados = registros.copy()
    df = pd.DataFrame(registros)
    frames = df['frame'].unique()

    for frame_id in frames:
        puntos_frame = df[df['frame'] == frame_id]

        # Punto medio de hombros
        hombro_izq = puntos_frame[puntos_frame['marcador'] == 'hombro_izquierdo']
        hombro_der = puntos_frame[puntos_frame['marcador'] == 'hombro_derecho']

        if not hombro_izq.empty and not hombro_der.empty:
            x1, y1 = hombro_izq.iloc[0]['X (px)'], hombro_izq.iloc[0]['Y (px)']
            x2, y2 = hombro_der.iloc[0]['X (px)'], hombro_der.iloc[0]['Y (px)']

            if pd.notnull(x1) and pd.notnull(y1) and pd.notnull(x2) and pd.notnull(y2):
                x_medio = (x1 + x2) / 2
                y_medio = (y1 + y2) / 2
            else:
                x_medio = None
                y_medio = None
        else:
            x_medio = None
            y_medio = None

        registros_actualizados.append({
            'marcador': 'punto_medio_hombros',
            'frame': frame_id,
            'X (px)': x_medio,
            'Y (px)': y_medio
        })

        # Punto medio de caderas
        cadera_izq = puntos_frame[puntos_frame['marcador'] == 'cadera_izquierda']
        cadera_der = puntos_frame[puntos_frame['marcador'] == 'cadera_derecha']

        if not cadera_izq.empty and not cadera_der.empty:
            x1, y1 = cadera_izq.iloc[0]['X (px)'], cadera_izq.iloc[0]['Y (px)']
            x2, y2 = cadera_der.iloc[0]['X (px)'], cadera_der.iloc[0]['Y (px)']

            if pd.notnull(x1) and pd.notnull(y1) and pd.notnull(x2) and pd.notnull(y2):
                x_medio = (x1 + x2) / 2
                y_medio = (y1 + y2) / 2
            else:
                x_medio = None
                y_medio = None
        else:
            x_medio = None
            y_medio = None

        registros_actualizados.append({
            'marcador': 'punto_medio_caderas',
            'frame': frame_id,
            'X (px)': x_medio,
            'Y (px)': y_medio
        })

    return registros_actualizados


def lectura_auxiliar(path):
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
                        visibilidad(landmarks,mp_pose)

                        if verificar_cambios(landmarks, mp_pose, historial_tobillo_pateo, historial_tobillo_no_pateo):
                            print("Recalculando")
                            continue

                        if inicio_video(landmarks,mp_pose) and visibilidad(landmarks,mp_pose):
                            bandera_rodilla = True
                    
                        #if camara == "T":
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

def lectura_video(path, camara):
    frames_seguidos = deque(maxlen=3)
    historial_tobillo_pateo = []
    historial_tobillo_no_pateo = []

    mp_marcar = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    frame_corte_final = None
    mejor_distancia = float('-inf')
    num_frame = 0
    index_frame = 0

    captura = cv2.VideoCapture(path, cv2.CAP_FFMPEG)
    total_frames = int(captura.get(cv2.CAP_PROP_FRAME_COUNT))  
    frame_final = int(captura.get(cv2.CAP_PROP_FRAME_COUNT))

    with mp_pose.Pose(min_detection_confidence=0.80, min_tracking_confidence=0.90, model_complexity=2) as pose: 
        while captura.isOpened():
            
            ret, frame = captura.read()
            if not ret:
                break
            else: 
                index_frame += 1
                if camara == "trasera":
                    frame = cv2.rotate(frame, cv2.ROTATE_180) 
                elif camara == "izquierda":
                    frame = cv2.rotate(frame, cv2.ROTATE_180)

                resultados = pose.process(frame)

                if resultados.pose_landmarks:
                    landmarks = resultados.pose_landmarks.landmark

                    if verificar_cambios(landmarks, mp_pose, historial_tobillo_pateo, historial_tobillo_no_pateo):
                        continue

                    distancia = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y - landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y
                    frames_seguidos.append(distancia > mejor_distancia and derecha_atras(landmarks,mp_pose, camara) and visibilidad(landmarks, mp_pose))

                    if (distancia > mejor_distancia and derecha_atras(landmarks,mp_pose, camara) and visibilidad(landmarks, mp_pose)):
                        mejor_distancia = distancia
                        if all(frames_seguidos):
                            num_frame = index_frame
            
            if index_frame >= total_frames:
                break

    captura.release()

    frames_analizados = 0
    historial_tobillo_pateo = []
    historial_tobillo_no_pateo = []

    if num_frame != 0:
        i = 0

        captura_final = cv2.VideoCapture(path, cv2.CAP_FFMPEG)

        with mp_pose.Pose(min_detection_confidence=0.85, min_tracking_confidence=0.90, model_complexity=2) as pose_final:
            while captura_final.isOpened():
                ret, frame = captura_final.read()
                if not ret:
                    break

                i += 1

                if camara == "trasera":
                    frame = cv2.rotate(frame, cv2.ROTATE_180) 
                elif camara == "izquierda":
                    frame = cv2.rotate(frame, cv2.ROTATE_180)

                resultados = pose_final.process(frame)

                if resultados.pose_landmarks:
                    landmarks = resultados.pose_landmarks.landmark

                    if verificar_cambios(landmarks, mp_pose, historial_tobillo_pateo, historial_tobillo_no_pateo):
                        continue

                    if fin_video_coordenadas(camara, landmarks, mp_pose, True) and not derecha_atras(landmarks, mp_pose, camara) and i >= num_frame:
                        frame_corte_final = i
                        break
    

                if i >= total_frames:
                    break

        captura_final.release()

    return(num_frame, (frame_corte_final), (num_frame!=0), frame_corte_final!=None)

def extraer_landmarks(path, frame_inicio, frame_final, camara, puntos_a_usar):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2, 
        min_detection_confidence=0.8,
        min_tracking_confidence=0.85
    )

    historial_tobillo_pateo = []
    historial_tobillo_no_pateo = []

    cap = cv2.VideoCapture(path,cv2.CAP_FFMPEG)
    ancho = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    alto = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_id = 0

    registros_video = []
    landmarks_anterior = None
    
    for i in range(frame_inicio, frame_final + 1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()

        if not ret:
            break

        if camara == "trasera":
            frame = cv2.rotate(frame, cv2.ROTATE_180)  
        elif camara == "izquierda":
            frame = cv2.rotate(frame, cv2.ROTATE_180)  

        resultados = pose.process(frame)

        if resultados.pose_landmarks:
            landmarks = resultados.pose_landmarks.landmark

            if frame_id >= frame_inicio and verificar_cambios_piernas(landmarks, mp_pose, historial_tobillo_pateo, historial_tobillo_no_pateo):
                if landmarks_anterior is not None:
                    landmarks = deepcopy(landmarks_anterior)
            else:
                landmarks_anterior = deepcopy(landmarks)

            for marcador, idx in puntos_a_usar.items():
                try:
                    punto = landmarks[idx]
                    x = punto.x * ancho
                    y = punto.y * alto
                except:
                    x = None
                    y = None
                registros_video.append({
                    'marcador': marcador,
                    'frame': frame_id,
                    'X (px)': x,
                    'Y (px)': y
                })

            landmarks_anterior = deepcopy(landmarks)

        else:
            for marcador in puntos_a_usar:
                registros_video.append({
                    'marcador': marcador,
                    'frame': frame_id,
                    'X (px)': None,
                    'Y (px)': None
                })

        frame_id += 1
     
    df_distancias = data_frame_distancias(registros_video)
    registros_video = corregir_primer_frame(df_distancias, registros_video,puntos_a_usar)
    registros_video = corregir_frames_estaticos(df_distancias, registros_video, puntos_a_usar)
    df_distancias = data_frame_distancias(registros_video)
    registros_video = corregir_frames_grandes(df_distancias, registros_video, puntos_a_usar, camara)
    df_distancias = data_frame_distancias(registros_video)
    registros_video = corregir_frames_grandes(df_distancias, registros_video, puntos_a_usar, camara)
    df_distancias = data_frame_distancias(registros_video)
    registros_video = corregir_frames_estaticos(df_distancias, registros_video, puntos_a_usar)
    df_distancias = data_frame_distancias(registros_video)

    cap.release()
    return registros_video


def cortar_video(path, path_destino, frame_inicio, frame_final, camara, registros, conexiones):
    captura = cv2.VideoCapture(path, cv2.CAP_FFMPEG)
    path_destino = path_destino.replace('.MOV','.mp4')

    if not captura.isOpened():
        return

    fps = int(captura.get(cv2.CAP_PROP_FPS))  
    width = int(captura.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(captura.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    total_frames = int(captura.get(cv2.CAP_PROP_FRAME_COUNT))  
    writer = cv2.VideoWriter(path_destino, fourcc, 5, (width, height))

    captura.set(cv2.CAP_PROP_POS_FRAMES, frame_inicio)

    angulos_por_frame = [] 

    momento = 0
    referencia_final = frame_final - frame_inicio
    for i in range(frame_inicio, frame_final + 1):
        ret, frame = captura.read()
        if not ret:
            break
        if camara == "trasera":
            frame = cv2.rotate(frame, cv2.ROTATE_180)  
        elif camara == "izquierda":
            frame = cv2.rotate(frame, cv2.ROTATE_180)  

        puntos_frame = [p for p in registros if p["frame"] == i - frame_inicio]
        coords = {}

        for punto in puntos_frame:
            marcador = punto["marcador"]
            x = punto["X (px)"]
            y = punto["Y (px)"]

            if x is not None and y is not None:
                coords[punto["marcador"]] = (int(x), int(y))
                if not ((marcador in ["hombro_derecho", "hombro_izquierdo", "cadera_derecha", "cadera_izquierda"]) and camara == "trasera"):
                    cv2.circle(frame, (int(x), int(y)), 6, (128, 0, 255), -1)

        for a, b in conexiones:
            if a in coords and b in coords:
                cv2.line(frame, coords[a], coords[b], (255, 255, 0), 3)

        datos_frame = {'frame': i}
        print(momento, " - ", frame_final)
        if camara == "derecha":
            if all(k in coords for k in ["cadera_derecha", "rodilla_derecha", "tobillo_derecho"]):
                ang_rodilla = calcular_angulo(coords["cadera_derecha"], coords["rodilla_derecha"], coords["tobillo_derecho"])
                datos_frame['angulo_rodilla'] = ang_rodilla
                if (momento == referencia_final - 1):
                    if((180-ang_rodilla) >= 31.6 and (180-ang_rodilla) <= 39.4):
                        cv2.putText(frame, f"{ang_rodilla}", coords["rodilla_derecha"], cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
                    elif(((180-ang_rodilla) >= 27.7 and (180-ang_rodilla) < 31.6) or ((180-ang_rodilla)>39.4 and (180-ang_rodilla) <= 43.3)):
                        cv2.putText(frame, f"{ang_rodilla}", coords["rodilla_derecha"], cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 2)
                    else: 
                        cv2.putText(frame, f"{ang_rodilla}", coords["rodilla_derecha"], cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
                else:
                     cv2.putText(frame, f"{ang_rodilla}", coords["rodilla_derecha"], cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 2)

            if all(k in coords for k in ["tobillo_derecho", "talon_derecho", "punta_derecha"]):
                ang_pie = calcular_angulo(coords["talon_derecho"], coords["tobillo_derecho"], coords["punta_derecha"])
                datos_frame['angulo_pie'] = ang_pie
                cv2.putText(frame, f"{ang_pie}", coords["tobillo_derecho"], cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 2)

            if all(k in coords for k in ["rodilla_derecha", "cadera_derecha", "hombro_derecho"]):
                ang_cadera = calcular_angulo(coords["hombro_derecho"], coords["cadera_derecha"], coords["rodilla_derecha"])
                datos_frame['angulo_cadera'] = ang_cadera
                if (momento == referencia_final - 1):
                    if((180-ang_cadera) >= 52.3 and (180-ang_cadera) <= 58.9):
                        cv2.putText(frame, f"{ang_cadera}", coords["cadera_derecha"], cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
                    elif(((180-ang_cadera) >= 49 and (180-ang_cadera) < 52.3) or ((180-ang_rodilla) > 58.9 and (180-ang_rodilla) <= 62.2)):
                        cv2.putText(frame, f"{ang_cadera}", coords["cadera_derecha"], cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 2)
                    else: 
                        cv2.putText(frame, f"{ang_cadera}", coords["cadera_derecha"], cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
                else:
                     cv2.putText(frame, f"{ang_cadera}", coords["cadera_derecha"], cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 2)
                

        elif camara == "izquierda":
            if all(k in coords for k in ["cadera_izquierda", "rodilla_izquierda", "tobillo_izquierdo"]):
                ang_rodilla = calcular_angulo(coords["cadera_izquierda"], coords["rodilla_izquierda"], coords["tobillo_izquierdo"])
                datos_frame['angulo_rodilla'] = ang_rodilla
                if (momento == referencia_final - 1):
                    if(ang_rodilla>= 130.35 and ang_rodilla <= 146.65):
                        cv2.putText(frame, f"{ang_rodilla}", coords["rodilla_izquierda"], cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
                    elif((ang_rodilla >= 122.2 and ang_rodilla < 130.35) or (ang_rodilla > 146.65 and ang_rodilla <= 154.8)):
                        cv2.putText(frame, f"{ang_rodilla}", coords["rodilla_izquierda"], cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 2)
                    else: 
                        cv2.putText(frame, f"{ang_rodilla}", coords["rodilla_izquierda"], cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
                elif (momento == 0):
                    if(ang_rodilla>= 152.7 and ang_rodilla <= 161.2):
                        cv2.putText(frame, f"{ang_rodilla}", coords["rodilla_izquierda"], cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
                    elif((ang_rodilla >= 148.5 and ang_rodilla < 152.7) or (ang_rodilla > 161.2 and ang_rodilla <= 165.5)):
                        cv2.putText(frame, f"{ang_rodilla}", coords["rodilla_izquierda"], cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 2)
                    else: 
                        cv2.putText(frame, f"{ang_rodilla}", coords["rodilla_izquierda"], cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
                else:
                    cv2.putText(frame, f"{ang_rodilla}", coords["rodilla_izquierda"], cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 2)

            if all(k in coords for k in ["tobillo_izquierdo", "talon_izquierdo", "punta_izquierda"]):
                ang_pie = calcular_angulo(coords["talon_izquierdo"], coords["tobillo_izquierdo"], coords["punta_izquierda"])
                datos_frame['angulo_pie'] = ang_pie
                cv2.putText(frame, f"{ang_pie}", coords["tobillo_izquierdo"], cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,0 ,0 ), 2)

            if all(k in coords for k in ["cadera_izquierda", "hombro_izquierdo"]):
                ang_vertical = calcular_angulo_vertical(coords["hombro_izquierdo"], coords["cadera_izquierda"])
                datos_frame['angulo_vertical_tronco'] = ang_vertical
                if (momento == referencia_final - 1):
                    if(ang_vertical>= 3.5 and ang_vertical <= 13.5):
                        cv2.putText(frame, f"{ang_vertical}", coords["cadera_izquierda"], cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
                    elif((ang_vertical >= -1 and ang_vertical < 3.5) or (ang_vertical > 13.5 and ang_vertical <= 18)):
                        cv2.putText(frame, f"{ang_vertical}", coords["cadera_izquierda"], cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 2)
                    else: 
                        cv2.putText(frame, f"{ang_vertical}", coords["cadera_izquierda"], cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
                elif (momento == 0):
                    if(ang_vertical>= -13.2 and ang_vertical <= -7.8):
                        cv2.putText(frame, f"{ang_vertical}", coords["cadera_izquierda"], cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
                    elif((ang_vertical >= -15.9 and ang_vertical < -13.2) or (ang_vertical > -7.8 and ang_vertical <= -5.1)):
                        cv2.putText(frame, f"{ang_vertical}", coords["cadera_izquierda"], cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 2)
                    else: 
                        cv2.putText(frame, f"{ang_vertical}", coords["cadera_izquierda"], cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
                else:
                    cv2.putText(frame, f"{ang_vertical}", coords["cadera_izquierda"], cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 2)
                

        elif camara == "trasera":
            if all(k in coords for k in ["punto_medio_hombros", "punto_medio_caderas", "talon_izquierdo"]):
                ang_inclinacion = calcular_angulo_inclinacion_tronco(coords["punto_medio_hombros"], coords["punto_medio_caderas"], coords["talon_izquierdo"])
                datos_frame['angulo_inclinacion_tronco'] = ang_inclinacion
                if (momento == referencia_final - 1):
                    if(ang_inclinacion >= 1.5 and ang_inclinacion <= 11.5):
                        cv2.putText(frame, f"{ang_inclinacion}", coords["punto_medio_caderas"], cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
                    elif((ang_inclinacion >= -3.5 and ang_inclinacion < 1.5) or (ang_inclinacion > 11.5 and ang_inclinacion <= 16.5)):
                        cv2.putText(frame, f"{ang_inclinacion}", coords["punto_medio_caderas"], cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 2)
                    else: 
                        cv2.putText(frame, f"{ang_inclinacion}", coords["punto_medio_caderas"], cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
                else:
                     cv2.putText(frame, f"{ang_inclinacion}", coords["punto_medio_caderas"], cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 2)
                

        angulos_por_frame.append(datos_frame) 

        cv2.imshow("hola", frame)
        cv2.waitKey(1)
        writer.write(frame)
        momento+=1

    captura.release()
    writer.release()
    cv2.destroyAllWindows()

    df_angulos = pd.DataFrame(angulos_por_frame)
    if camara == "derecha":
        global df_derecha
        df_derecha = df_angulos
    elif camara == "izquierda":
        global df_izquierda
        df_izquierda = df_angulos
    elif camara == "trasera":
        global df_trasera
        df_trasera = df_angulos

def index(request):
    # Cuando se recibe el formulario
    if request.method == "POST" and request.FILES:
        # Obtener los videos de la solicitud
        videoLD = request.FILES.get('videoLD')
        videoLI = request.FILES.get('videoLI')
        videoT = request.FILES.get('videoT')

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        fs = FileSystemStorage()

        # Crear nombres únicos para cada video
        videoLD_name = f"videoLD_{timestamp}.mp4"
        videoLI_name = f"videoLI_{timestamp}.mp4"
        videoT_name  = f"videoT_{timestamp}.mp4"

        # Guardar los archivos en el directorio adecuado y obtener sus URLs
        videoLD_path = fs.save(videoLD_name, videoLD)
        videoLI_path = fs.save(videoLI_name, videoLI)
        videoT_path = fs.save(videoT_name, videoT)

        abs_path_LD = fs.path(videoLD_path)
        abs_path_LI = fs.path(videoLI_path)
        abs_path_T  = fs.path(videoT_path)

        cropped_LD_path = fs.path(f"corte_{videoLD_name}")
        cropped_LI_path = fs.path(f"corte_{videoLI_name}")
        cropped_T_path  = fs.path(f"corte_{videoT_name}")

        if os.path.exists(abs_path_LD):
            puntos_derechos = {
                'punta_derecha': 32,
                'talon_derecho': 30,
                'tobillo_derecho': 28,
                'rodilla_derecha': 26,
                'cadera_derecha': 24,
                'hombro_derecho': 12
            }
            conexiones_derechos = [
                ('cadera_derecha', 'rodilla_derecha'),
                ('rodilla_derecha', 'tobillo_derecho'),
                ('tobillo_derecho', 'talon_derecho'),
                ('talon_derecho', 'punta_derecha'),
                ('cadera_derecha', 'hombro_derecho')
            ]
            inicio_derecha, final_derecha, bool_inicio_derecha, bool_final_derecha = lectura_video(abs_path_LD, "derecha")
            print("Okay derecha")
            if not bool_inicio_derecha or not bool_final_derecha:
                inicio_derecha, final_derecha, bool_inicio_derecha, bool_final_derecha = lectura_auxiliar(abs_path_LD, "derecha")
            
            if(bool_inicio_derecha and bool_final_derecha):
                registros_derecha = extraer_landmarks(abs_path_LD, inicio_derecha, final_derecha, "derecha",puntos_derechos)
                valor_ref = final_derecha - inicio_derecha
                cortar_video(abs_path_LD, cropped_LD_path, inicio_derecha, final_derecha, "derecha", registros_derecha, conexiones_derechos)
            print("Okay corte derecha")

        if os.path.exists(abs_path_LI):
            puntos_izquierdos = {
                'punta_izquierda': 31,
                'talon_izquierdo': 29,
                'tobillo_izquierdo': 27,
                'rodilla_izquierda': 25,
                'cadera_izquierda': 23,
                'hombro_izquierdo': 11
            }
            conexiones_izquierda = [
                ('cadera_izquierda', 'rodilla_izquierda'),
                ('rodilla_izquierda', 'tobillo_izquierdo'),
                ('tobillo_izquierdo', 'talon_izquierdo'),
                ('talon_izquierdo', 'punta_izquierda'),
                ('hombro_izquierdo', 'cadera_izquierda')
            ]

            inicio_izquierda, final_izquierda, bool_inicio_izquierda, bool_final_izquierda = lectura_video(abs_path_LI, "izquierda")
            if not bool_inicio_izquierda or not bool_final_izquierda: 
                inicio_izquierda, final_izquierda, bool_inicio_izquierda, bool_final_izquierda = lectura_auxiliar(abs_path_LI, "izquierda")
                print("Okay izquierda")

            if(bool_inicio_izquierda and bool_final_izquierda):
                registros_izquierda = extraer_landmarks(abs_path_LI, inicio_izquierda, (inicio_izquierda + valor_ref), "izquierda",puntos_izquierdos)
                cortar_video(abs_path_LI, cropped_LI_path, inicio_izquierda, inicio_izquierda + valor_ref, "izquierda", registros_izquierda, conexiones_izquierda)
                print("Okay corte izquierda")
        if os.path.exists(abs_path_T):
            puntos_traseros = {
                'hombro_izquierdo': 11,
                'hombro_derecho': 12,
                'cadera_izquierda': 23,
                'cadera_derecha': 24,
                'talon_izquierdo': 29
            }
            conexiones_traseras = [
                ('punto_medio_hombros', 'punto_medio_caderas'),
                ('punto_medio_caderas', 'talon_izquierdo')
            ]

            inicio_trasera, final_trasera, bool_inicio_trasera, bool_final_trasera = lectura_video(abs_path_T, "trasera")
            print("Okay trasera")        
            if not bool_inicio_trasera or not bool_final_trasera: 
                inicio_trasera, final_trasera, bool_inicio_trasera, bool_final_trasera = lectura_auxiliar(abs_path_T, "trasera")
            
            if(bool_inicio_trasera and bool_final_trasera):
                registros_trasera = extraer_landmarks(abs_path_T, inicio_trasera, final_trasera, "trasera",puntos_traseros)
                print(registros_trasera)
                registros_trasera = calcular_puntos_medios(registros_trasera, puntos_traseros)
                print(registros_trasera)
                cortar_video(abs_path_T, cropped_T_path, inicio_trasera,inicio_trasera + valor_ref, "trasera", registros_trasera, conexiones_traseras)
                print("Okay corte trasera")

        videoLD_url = fs.url(f"corte_{videoLD_name}")
        videoLI_url = fs.url(f"corte_{videoLI_name}")
        videoT_url  = fs.url(f"corte_{videoT_name}")

        print("URL videoLD:", videoLD_url)
        print("URL videoLI:", videoLI_url)
        print("URL videoT:", videoT_url)

        return render(request, 'index.html', {
            'videoLD_url': videoLD_url,
            'videoLI_url': videoLI_url,
            'videoT_url': videoT_url
        })

    return render(request, 'index.html')


def descargar_excel(request):
    global df_derecha, df_izquierda, df_trasera

    if df_derecha is None and df_izquierda is None and df_trasera is None:
        return HttpResponse("No hay datos para exportar", status=400)

    response = HttpResponse(
        content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )
    response['Content-Disposition'] = 'attachment; filename="angulos_biomecanicos.xlsx"'

    with pd.ExcelWriter(response, engine='openpyxl') as writer:
        if df_derecha is not None:
            df_derecha.to_excel(writer, sheet_name='Vista Derecha', index=False)
        if df_izquierda is not None:
            df_izquierda.to_excel(writer, sheet_name='Vista Izquierda', index=False)
        if df_trasera is not None:
            df_trasera.to_excel(writer, sheet_name='Vista Trasera', index=False)

    return response
