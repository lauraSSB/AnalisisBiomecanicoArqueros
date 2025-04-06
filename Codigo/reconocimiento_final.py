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
lado = 'LI'  # Cambiar a 'LD' si es lado derecho
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

if lado == 'LI':
    puntos_a_usar = puntos_izquierdos
    indices_dibujo = [23, 25, 27, 29, 31]
    conexiones = [(23, 25), (25, 27), (27, 29), (29, 31)]
    orden_pierna = list(puntos_izquierdos.keys())
    titulo_pierna = "Pierna izquierda"
else:
    puntos_a_usar = puntos_derechos
    indices_dibujo = [24, 26, 28, 30, 32]
    conexiones = [(24, 26), (26, 28), (28, 30), (30, 32)]
    orden_pierna = list(puntos_derechos.keys())
    titulo_pierna = "Pierna derecha"

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


def verificar_cambios_piernas(landmarks, mp_pose, historial_tobillo_pateo, historial_tobillo_no_pateo):
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
        if cambio_pateo > 0.05 or cambio_no_pateo > 0.05:
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
    elif lado == "LI":
        ax.set_xlim(400, 1200)

    ax.set_ylim(400, 1000)
    ax.invert_yaxis()
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(f"{titulo_pierna} - Archivo {archivo_num}, Frame {frame_id}")
    ax.set_xlabel("X (px)")
    ax.set_ylabel("Y (px)")
    ax.grid(True)
    plt.tight_layout()
    plt.pause(0.5)
    plt.close()


def corregir_primer_frame(df_distancias, registros, umbral=15):
    registros_corregidos = deepcopy(registros)
    grupo_pierna = list(puntos_a_usar.keys())

    for archivo, grupo_archivo in df_distancias.groupby('archivo'):
        for marcador in grupo_pierna:
            grupo = grupo_archivo[grupo_archivo['marcador'] == marcador].sort_values('frame_anterior')

            if len(grupo) >= 3:
                d01 = grupo.iloc[0]['distancia']
                d12 = grupo.iloc[1]['distancia']
                d23 = grupo.iloc[2]['distancia']

                if d01 > umbral and d12 < umbral and d23 < umbral:
                    print("Corrige primer frame")
                    for m in grupo_pierna:
                        frame1 = next((r for r in registros_corregidos if r['archivo'] == archivo and r['marcador'] == m and r['frame'] == 1), None)
                        frame0 = next((r for r in registros_corregidos if r['archivo'] == archivo and r['marcador'] == m and r['frame'] == 0), None)
                        if frame0 and frame1:
                            frame0['X (px)'] = frame1['X (px)']
                            frame0['Y (px)'] = frame1['Y (px)']
                    break

    return registros_corregidos

def corregir_frames_estaticos(df_distancias, registros_video, puntos_a_usar):
    registros_corregidos = deepcopy(registros_video)
    grupo_pierna = list(puntos_a_usar.keys())

    df = pd.DataFrame(registros_video)

    for frame in df_distancias['frame_actual'].unique():
        grupo_frame = df_distancias[df_distancias['frame_actual'] == frame]

        if set(grupo_frame['marcador']) == set(grupo_pierna) and all(grupo_frame['distancia'] == 0):
            print(f"\nCorrigiendo frame {frame} (todas las distancias 0)")

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

videos = []
excluidos = [3, 4, 7, 10, 11, 12, 44, 61, 67, 69, 75, 76, 77, 79, 80, 81, 82]
for i in range(45, 46):
    if i not in excluidos:
        nombre = f'Piso_{lado}_{i}.mp4'
        ruta = os.path.join(carpeta_videos, nombre)
        if os.path.exists(ruta):
            videos.append((ruta, i))
        else:
            print(f"No se encontró el video: {ruta}")

registros = []

for ruta_video, numero_archivo in videos:
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,
        min_detection_confidence=0.85,
        min_tracking_confidence=0.9
    )

    historial_tobillo_pateo = []
    historial_tobillo_no_pateo = []

    cap = cv2.VideoCapture(ruta_video)
    ancho = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    alto = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_id = 10

    print(f"\nProcesando {os.path.basename(ruta_video)}, con {total_frames} frames")

    registros_video = []
    landmarks_anterior = None

    while frame_id <= total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (int(frame.shape[1] * 0.6), int(frame.shape[0] * 0.6)))
        resultados = pose.process(frame)

        if resultados.pose_landmarks:
            landmarks = resultados.pose_landmarks.landmark

            if frame_id >= 1 and verificar_cambios_piernas(landmarks, mp_pose, historial_tobillo_pateo, historial_tobillo_no_pateo):
                if landmarks_anterior is not None:
                    print("Corrige cambio piernas")
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

            for idx in indices_dibujo:
                punto = landmarks[idx]
                x = int(punto.x * frame.shape[1])
                y = int(punto.y * frame.shape[0])
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

            for a, b in conexiones:
                punto1 = landmarks[a]
                punto2 = landmarks[b]
                x1, y1 = int(punto1.x * frame.shape[1]), int(punto1.y * frame.shape[0])
                x2, y2 = int(punto2.x * frame.shape[1]), int(punto2.y * frame.shape[0])
                cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

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

        cv2.imshow(f'MediaPipe - Visualización {titulo_pierna}', frame)
        key = cv2.waitKey(1)
        if key == 27:
            break
        frame_id += 1

        time.sleep(15)

    df_distancias = data_frame_distancias(registros_video)
    registros_video = corregir_primer_frame(df_distancias, registros_video)

    registros_video = corregir_frames_estaticos(df_distancias, registros_video, puntos_a_usar)


    registros.extend(registros_video)
    # for i in range(0, frame_id):
    #     graficar_puntos_xy(registros_video, numero_archivo, i)

    cap.release()
    cv2.destroyAllWindows()

df_final = pd.DataFrame(registros)
#df_final.to_csv(output_csv, index=False)
