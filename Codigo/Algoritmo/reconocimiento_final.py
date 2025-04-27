import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from copy import deepcopy
import os
import time

# --- Parámetros Globales ---
MAX_HISTORIAL = 5
carpeta_videos = 'G:/Mi unidad/Videos Trabajo de Grado/Lateral Izquierda Limpia 2'
lado = 'LI'  # 'LI', 'LD' o 'T'
output_csv = f'G:/Mi unidad/Videos Trabajo de Grado/puntos_clave255_{lado}.csv'

# --- Configuración de puntos y conexiones ---
if lado == 'LI':
    puntos_a_usar = {
        'punta_izquierda': 31,
        'talon_izquierdo': 29,
        'tobillo_izquierdo': 27,
        'rodilla_izquierda': 25,
        'cadera_izquierda': 23
    }
    conexiones = [(23, 25), (25, 27), (27, 29), (29, 31)]
elif lado == 'LD':
    puntos_a_usar = {
        'punta_derecha': 32,
        'talon_derecho': 30,
        'tobillo_derecho': 28,
        'rodilla_derecha': 26,
        'cadera_derecha': 24
    }
    conexiones = [(24, 26), (26, 28), (28, 30), (30, 32)]
elif lado == 'T':
    puntos_a_usar = {
        'cadera_izquierda': 23,
        'rodilla_izquierda': 25,
        'tobillo_izquierdo': 27,
        'talon_izquierdo': 29,
        'cadera_derecha': 24
    }
    conexiones = [(23, 25), (25, 27), (27, 29)]
else:
    raise ValueError("Lado no reconocido. Usa 'LI', 'LD' o 'T'")

mp_pose = mp.solutions.pose

# --- Funciones auxiliares ---

def verificar_cambios_piernas(landmarks, historial_pateo, historial_no_pateo):
    tobillo_pateo = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x
    tobillo_no_pateo = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x
    historial_pateo.append(tobillo_pateo)
    historial_no_pateo.append(tobillo_no_pateo)
    if len(historial_pateo) > MAX_HISTORIAL:
        historial_pateo.pop(0)
        historial_no_pateo.pop(0)
    if len(historial_pateo) >= 2:
        cambio_pateo = abs(historial_pateo[-1] - historial_pateo[-2])
        cambio_no_pateo = abs(historial_no_pateo[-1] - historial_no_pateo[-2])
        return cambio_pateo > 0.05 or cambio_no_pateo > 0.05
    return False
    
def extraer_landmarks(path, numero_archivo):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,
        min_detection_confidence=0.8,
        min_tracking_confidence=0.85
    )

    historial_tobillo_pateo = []
    historial_tobillo_no_pateo = []

    cap = cv2.VideoCapture(path, cv2.CAP_FFMPEG)
    ancho = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    alto = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_id = 10  # empiezas en 10 como hacías antes
    registros_video = []
    landmarks_anterior = None

    while frame_id <= total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = cap.read()

        if not ret:
            break

        resultados = pose.process(frame)

        if resultados.pose_landmarks:
            landmarks = resultados.pose_landmarks.landmark

            if verificar_cambios_piernas(landmarks, historial_tobillo_pateo, historial_tobillo_no_pateo):
                if landmarks_anterior is not None:
                    landmarks = deepcopy(landmarks_anterior)
            else:
                landmarks_anterior = deepcopy(landmarks)

            for marcador, idx in puntos_a_usar.items():
                punto = landmarks[idx]
                registros_video.append({
                    'archivo': numero_archivo,
                    'marcador': marcador,
                    'frame': frame_id,
                    'X (px)': punto.x * ancho,
                    'Y (px)': punto.y * alto
                })

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

    cap.release()
    cv2.destroyAllWindows()
    return registros_video

def data_frame_distancias(registros):
    df = pd.DataFrame(registros)
    distancias = []
    for (archivo, marcador), grupo in df.groupby(['archivo', 'marcador']):
        grupo = grupo.sort_values('frame')
        for i in range(1, len(grupo)):
            fila_anterior = grupo.iloc[i - 1]
            fila_actual = grupo.iloc[i]
            dx = fila_actual['X (px)'] - fila_anterior['X (px)']
            dy = fila_actual['Y (px)'] - fila_anterior['Y (px)']
            distancia = np.sqrt(dx**2 + dy**2)
            distancias.append({
                'archivo': archivo,
                'marcador': marcador,
                'frame_anterior': fila_anterior['frame'],
                'frame_actual': fila_actual['frame'],
                'distancia': distancia
            })
    return pd.DataFrame(distancias)

def corregir_primer_frame(df_distancias, registros, umbral=15):
    registros_corregidos = deepcopy(registros)
    grupo_pierna = list(puntos_a_usar.keys())

    grupo_archivo = df_distancias 

    for marcador in grupo_pierna:
        grupo = grupo_archivo[grupo_archivo['marcador'] == marcador].sort_values('frame_anterior')

        if len(grupo) >= 3:
            d01 = grupo.iloc[0]['distancia']
            d12 = grupo.iloc[1]['distancia']
            d23 = grupo.iloc[2]['distancia']

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
            for marcador in grupo_pierna:
                anterior = next((r for r in registros_corregidos if r['marcador'] == marcador and r['frame'] == frame - 1), None)
                posterior = next((r for r in registros_corregidos if r['marcador'] == marcador and r['frame'] == frame + 1), None)
                if anterior and posterior:
                    promedio_x = (anterior['X (px)'] + posterior['X (px)']) / 2
                    promedio_y = (anterior['Y (px)'] + posterior['Y (px)']) / 2
                    punto_actual = next((r for r in registros_corregidos if r['marcador'] == marcador and r['frame'] == frame), None)
                    if punto_actual:
                        punto_actual['X (px)'] = promedio_x
                        punto_actual['Y (px)'] = promedio_y
    return registros_corregidos

def corregir_frames_grandes(df_distancias, registros_video, lado):
    registros_corregidos = deepcopy(registros_video)
    df = pd.DataFrame(registros_corregidos)

    if lado == 'T':
        umbrales_por_marcador = {
            'cadera_izquierda': 10,
            'rodilla_izquierda': 18,
            'tobillo_izquierdo': 10,
            'talon_izquierdo': 10,
            'cadera_derecha': 10
        }
    elif lado == 'LI':
        umbrales_por_marcador = {
            'cadera_izquierda': 22,
            'rodilla_izquierda': 60,
            'tobillo_izquierdo': 130,
            'talon_izquierdo': 150,
            'punta_izquierda': 150
        }
    elif lado == 'LD':
        umbrales_por_marcador = {
            'cadera_derecha': 22,
            'rodilla_derecha': 60,
            'tobillo_derecho': 130,
            'talon_derecho': 150,
            'punta_derecha': 150
        }
    else:
        umbrales_por_marcador = {}

    for _, fila in df_distancias.iterrows():
        marcador = fila['marcador']
        distancia = fila['distancia']
        frame_actual = fila['frame_actual']
        frame_anterior = fila['frame_anterior']
        umbral = umbrales_por_marcador.get(marcador, 20)

        if distancia is not None and distancia > umbral:
            idx_actual = df[(df['marcador'] == marcador) & (df['frame'] == frame_actual)].index
            anterior = df[(df['marcador'] == marcador) & (df['frame'] == frame_anterior)]
            if not anterior.empty and not idx_actual.empty:
                x_ant, y_ant = anterior.iloc[0]['X (px)'], anterior.iloc[0]['Y (px)']
                df.loc[idx_actual, 'X (px)'] = x_ant
                df.loc[idx_actual, 'Y (px)'] = y_ant

    return df.to_dict('records')



invalidos = [3, 4, 7, 10, 11, 12, 30, 37, 44, 61, 67, 69, 75, 76, 77, 79, 80, 81, 82]

videos = []
for i in range(1, 100):
    if i not in invalidos:
        nombre = f'Piso_{lado}_{i}.mp4'
        ruta = os.path.join(carpeta_videos, nombre)
        if os.path.exists(ruta):
            videos.append((ruta, i))
        else:
            print(f"No se encontró el video: {ruta}")


registros = []
for ruta_video, numero_archivo in videos:
    registros_video = extraer_landmarks(ruta_video, numero_archivo)
    df_distancias = data_frame_distancias(registros_video)
    registros_video = corregir_primer_frame(df_distancias, registros_video)
    df_distancias = data_frame_distancias(registros_video)
    registros_video = corregir_frames_estaticos(df_distancias, registros_video, puntos_a_usar)
    df_distancias = data_frame_distancias(registros_video)
    registros_video = corregir_frames_grandes(df_distancias, registros_video, lado)
    registros.extend(registros_video)

# Exportar resultados
df_final = pd.DataFrame(registros)
df_final.to_csv(output_csv, index=False)