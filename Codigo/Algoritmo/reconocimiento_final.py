import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from copy import deepcopy
import os
import time
import matplotlib.pyplot as plt

MAX_HISTORIAL = 5

carpeta_videos = 'G:/Mi unidad/Videos Trabajo de Grado/Lateral Izquierda Limpia 2'
lado = 'LI' 
output_csv = f'G:/Mi unidad/Videos Trabajo de Grado/puntos_clave_{lado}.csv'

puntos_derechos = {
    'punta_derecha': 32,
    'talon_derecho': 30,
    'tobillo_derecho': 28,
    'rodilla_derecha': 26,
    'cadera_derecha': 24
}

puntos_izquierdos = {
    'punta_izquierda': 31,
    'talon_izquierdo': 29,
    'tobillo_izquierdo': 27,
    'rodilla_izquierda': 25,
    'cadera_izquierda': 23
}

puntos_traseros = {
    'cadera_derecha': 24,
    'cadera_izquierda': 23,
    'hombro_derecho': 12,
    'hombro_izquierdo': 11,
    'rodilla_izquierda': 25,
    'talon_izquierdo': 29,
    'tobillo_izquierdo': 27
}

if lado == 'LI':
    puntos_a_usar = puntos_izquierdos
    indices_dibujo = [23, 25, 27, 29, 31]
    conexiones = [(23, 25), (25, 27), (27, 29), (29, 31)]
    orden_pierna = list(puntos_izquierdos.keys())
    camara = "izquierda"
elif lado == 'LD':
    puntos_a_usar = puntos_derechos
    indices_dibujo = [24, 26, 28, 30, 32]
    conexiones = [(24, 26), (26, 28), (28, 30), (30, 32)]
    orden_pierna = list(puntos_derechos.keys())
    camara = "derecha"
else:
    puntos_a_usar = puntos_traseros
    indices_dibujo = [24, 23, 12, 11, 25, 27, 29]
    conexiones = [
        (24, 23),(24, 12),(23, 11),(23, 25),(25, 27),(27, 29)  
    ]
    orden_pierna = list(puntos_traseros.keys())
    camara = "trasera"

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

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

def verificar_cambios(landmarks, mp_pose,historial_tobillo_pateo,historial_tobillo_no_pateo):
    tobillo_pateo = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x
    tobillo_no_pateo = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x

    historial_tobillo_pateo.append(tobillo_pateo)
    historial_tobillo_no_pateo.append(tobillo_no_pateo)

    if len(historial_tobillo_pateo) > MAX_HISTORIAL:
        historial_tobillo_pateo.pop(0)
        historial_tobillo_no_pateo.pop(0)

    if len(historial_tobillo_pateo) >= 2:
        cambio_pateo = abs(historial_tobillo_pateo[-1] - historial_tobillo_pateo[-2])
        cambio_no_pateo = abs(historial_tobillo_no_pateo[-1] - historial_tobillo_no_pateo[-2])
        if cambio_pateo > 0.06 or cambio_no_pateo > 0.06:
            return True

    return False


def graficar_puntos_xy(registros, archivo_num, frame_id):
    puntos = [r for r in registros if r['archivo'] == archivo_num and r['frame'] == frame_id]

    coordenadas = {}
    for punto in puntos:
        x = punto['X (px)']
        y = punto['Y (px)']
        if x is not None and y is not None:
            coordenadas[punto['marcador']] = (x, y)

    fig, ax = plt.subplots(figsize=(6, 8))

    for marcador, (x, y) in coordenadas.items():
        ax.scatter(x, y, color='blue')
        ax.text(x + 5, y, marcador, fontsize=9)

    for a, b in zip(orden_pierna[:-1], orden_pierna[1:]):
        if a in coordenadas and b in coordenadas:
            x1, y1 = coordenadas[a]
            x2, y2 = coordenadas[b]
            ax.plot([x1, x2], [y1, y2], color='red', linewidth=2)

    if lado == "LD":
        ax.set_xlim(700, 1600)
    else:
        ax.set_xlim(400, 1200)

    ax.set_ylim(400, 1000)
    ax.invert_yaxis()
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(f"{camara} - Archivo {archivo_num}, Frame {frame_id}")
    ax.set_xlabel("X (px)")
    ax.set_ylabel("Y (px)")
    ax.grid(True)
    plt.tight_layout()
    plt.pause(0.5)
    plt.close()


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

def extraer_landmarks(path, puntos_a_usar, numero_archivo):
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
    
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        resultados = pose.process(frame)

        if resultados.pose_landmarks:
            landmarks = resultados.pose_landmarks.landmark

            if verificar_cambios(landmarks, mp_pose, historial_tobillo_pateo, historial_tobillo_no_pateo):
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
                    'archivo': numero_archivo,
                    'marcador': marcador,
                    'frame': frame_id,
                    'X (px)': x,
                    'Y (px)': y
                })

            landmarks_anterior = deepcopy(landmarks)

        else:
            for marcador in puntos_a_usar:
                registros_video.append({
                    'archivo': numero_archivo,
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

videos = []
excluidos = [3, 4, 7, 10, 11, 12, 44, 61, 67, 69, 75, 76, 77, 79, 80, 81, 82]
for i in range(1, 100):
    if i not in excluidos:
        nombre = f'Piso_{lado}_{i}.mp4'
        ruta = os.path.join(carpeta_videos, nombre)
        if os.path.exists(ruta):
            videos.append((ruta, i))
        else:
            print(f"No se encontró el video: {ruta}")

registros = []

for ruta_video, numero_archivo in videos:
    registros_video = extraer_landmarks(ruta_video,puntos_a_usar, numero_archivo)
    print(registros_video)
    registros.extend(registros_video)
    graficar_puntos_xy(registros, numero_archivo, 2)


df_final = pd.DataFrame(registros)
df_final.to_csv(output_csv, index=False)
