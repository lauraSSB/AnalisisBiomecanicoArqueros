import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

def calcular_angulo(p1, p2, p3):
    a = np.array([p1[0] - p2[0], p1[1] - p2[1]])
    b = np.array([p3[0] - p2[0], p3[1] - p2[1]])
    angulo_rad = math.acos(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    return round(np.degrees(angulo_rad),3)

def cargar_datos(path):
    return pd.read_excel(path) if path.endswith('.xlsx') else pd.read_csv(path)

def construir_diccionario(df):
    data = {}
    for (archivo, frame), grupo in df.groupby(['archivo', 'frame']):
        puntos = {}
        for _, fila in grupo.iterrows():
            puntos[fila['marcador']] = (fila['X (px)'], fila['Y (px)'])
        data[(archivo, frame)] = puntos
    return data

def transformar_kinovea(puntos):
    return {k: (962 + v[0], 540 - v[1]) for k, v in puntos.items()}

def graficar_comparacion(puntos_k, puntos_m, archivo, frame, lado):
    fig, ax = plt.subplots(figsize=(6, 8))

    if lado == "izquierda":
        conexiones = [
            ('cadera_izquierda', 'rodilla_izquierda'),
            ('rodilla_izquierda', 'tobillo_izquierdo'),
            ('tobillo_izquierdo', 'talon_izquierdo'),
            ('talon_izquierdo', 'punta_izquierda')
        ]
    else:
        conexiones = [
            ('cadera_derecha', 'rodilla_derecha'),
            ('rodilla_derecha', 'tobillo_derecho'),
            ('tobillo_derecho', 'talon_derecha'),
            ('talon_derecha', 'punta_derecha')
        ]

    puntos_k = transformar_kinovea(puntos_k)

    for marcador, (x, y) in puntos_k.items():
        ax.scatter(x, y, color='blue')
    for a, b in conexiones:
        if a in puntos_k and b in puntos_k:
            x1, y1 = puntos_k[a]
            x2, y2 = puntos_k[b]
            ax.plot([x1, x2], [y1, y2], color='blue', linestyle='-', linewidth=1)

    for marcador, (x, y) in puntos_m.items():
        ax.scatter(x, y, color='red')
    for a, b in conexiones:
        if a in puntos_m and b in puntos_m:
            x1, y1 = puntos_m[a]
            x2, y2 = puntos_m[b]
            ax.plot([x1, x2], [y1, y2], color='red', linestyle='--', linewidth=1)

    if lado == "izquierda":
        ax.set_xlim(400, 1200)
    else: 
        ax.set_xlim(800, 1600)

    ax.set_ylim(400, 1000)
    ax.invert_yaxis()
    ax.set_title(f"Comparación XY - Archivo {archivo} - Frame {frame}\nAzul: Kinovea | Rojo: MediaPipe")
    ax.set_xlabel("X (px)")
    ax.set_ylabel("Y (px)")
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True)
    plt.tight_layout()
    plt.pause(1)
    plt.close()

def comparar_angulos(path_kinovea, path_mediapipe):
    df_kinovea = cargar_datos(path_kinovea)
    df_mediapipe = cargar_datos(path_mediapipe)

    datos_kinovea = construir_diccionario(df_kinovea)
    datos_mediapipe = construir_diccionario(df_mediapipe)

    # Detectar si los datos son para pierna izquierda o derecha según el nombre del archivo
    if "LI" in path_kinovea.upper():
        lado = "izquierda"
        cadera = "cadera_izquierda"
        rodilla = "rodilla_izquierda"
        tobillo = "tobillo_izquierdo"
        talon = "talon_izquierdo"
        punta = "punta_izquierda"
    else:
        lado = "derecha"
        cadera = "cadera_derecha"
        rodilla = "rodilla_derecha"
        tobillo = "tobillo_derecho"
        talon = "talon_derecho"
        punta = "punta_derecha"

    resultados = []

    claves_comunes = set(datos_kinovea.keys()) & set(datos_mediapipe.keys())
    for archivo_frame in sorted(claves_comunes):
        archivo, frame = archivo_frame

        puntos_k = datos_kinovea[archivo_frame]
        puntos_m = datos_mediapipe[archivo_frame]

        try:
            angulo_rodilla_k = calcular_angulo(puntos_k[cadera], puntos_k[rodilla], puntos_k[tobillo])
            angulo_pie_k = calcular_angulo(puntos_k[tobillo], puntos_k[talon], puntos_k[punta])
        except Exception as e:
            angulo_rodilla_k = None
            angulo_pie_k = None

        try:
            angulo_rodilla_m = calcular_angulo(puntos_m[cadera], puntos_m[rodilla], puntos_m[tobillo])
            angulo_pie_m = calcular_angulo(puntos_m[tobillo], puntos_m[talon], puntos_m[punta])
        except Exception as e:
            angulo_rodilla_m = None
            angulo_pie_m = None

        graficar_comparacion(puntos_k, puntos_m, archivo, frame,lado)

        resultados.append({
            'archivo': archivo,
            'frame': frame,
            'angulo_rodilla_kinovea': angulo_rodilla_k,
            'angulo_rodilla_mediapipe': angulo_rodilla_m,
            'angulo_pie_kinovea': angulo_pie_k,
            'angulo_pie_mediapipe': angulo_pie_m
        })

    return pd.DataFrame(resultados)


# === USO DEL SCRIPT ===

path_kino = 'G:/Mi unidad/Videos Trabajo de Grado/Excel Kinovea LD/excel_unido.csv'
path_mediapipe = "G:/Mi unidad/Videos Trabajo de Grado/puntos_clave_LD.csv"

df_angulos = comparar_angulos(path_kino, path_mediapipe)

antes = len(df_angulos)
df_angulos = df_angulos.dropna()
despues = len(df_angulos)
print("Antes: ", antes, " Despues: ", despues)

df_angulos['diferencia1'] = round(df_angulos['angulo_rodilla_kinovea'] - df_angulos['angulo_rodilla_mediapipe'],3)
df_angulos['diferencia2'] = round(df_angulos['angulo_pie_kinovea'] - df_angulos['angulo_pie_mediapipe'],3)
#print(df_angulos)
df_angulos.to_csv("compaaLD.csv", index=False)
