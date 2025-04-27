import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import os
import time

#Función que calcula los ángulos en grados en 2D dados 3 puntos
def calcular_angulo(p1, p2, p3):
    a = np.array([p1[0] - p2[0], p1[1] - p2[1]])
    b = np.array([p3[0] - p2[0], p3[1] - p2[1]])
    angulo_rad = math.acos(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    return round(np.degrees(angulo_rad),3)

#Función que carga el excel y devuelve un dataframe
def cargar_datos(path):
    return pd.read_excel(path) if path.endswith('.xlsx') else pd.read_csv(path)

#Función que convierte un dataframe a diccionario
def construir_diccionario(df):
    data = {}
    for (archivo, frame), grupo in df.groupby(['archivo', 'frame']):
        puntos = {}
        for _, fila in grupo.iterrows():
            puntos[fila['marcador']] = (fila['X (px)'], fila['Y (px)'])
        data[(archivo, frame)] = puntos
    return data

#Función que convierte los puntos del sistema de referencia de Kinovea al sistema de referencia de Mediapipe
#Kinovea tiene el (0,0) en el centro de la imagen
#Mediapipe tiene el (0,0) en la esquina superior izquierda
#Todos los puntos se le suma 960 (la mitad de 1920) en X
#Todos los puntos se restan a 540 (la mitad de 1080) en Y
def transformar_kinovea(puntos):
    return {k: (960 + v[0], 540 - v[1]) for k, v in puntos.items()}


#Función que, dado los puntos, el archivo, el frame y el lado, gráfica en un plano (X,Y) los puntos de 
#Mediapipe y de Kinovea
def graficar_comparacion(puntos_k, puntos_m, archivo, frame, lado):
    fig, ax = plt.subplots(figsize=(6, 6))

    if lado == "izquierda":
        conexiones = [
            ('cadera_izquierda', 'rodilla_izquierda'),
            ('rodilla_izquierda', 'tobillo_izquierdo'),
            ('tobillo_izquierdo', 'talon_izquierdo'),
            ('talon_izquierdo', 'punta_izquierda')
        ]
    elif lado == "derecha":
        conexiones = [
            ('cadera_derecha', 'rodilla_derecha'),
            ('rodilla_derecha', 'tobillo_derecho'),
            ('tobillo_derecho', 'talon_derecha'),
            ('talon_derecha', 'punta_derecha')
        ]
    else:
        conexiones = [
            ('cadera_izquierda', 'rodilla_izquierda'),
            ('rodilla_izquierda', 'tobillo_izquierdo'),
            ('cadera_derecha','cadera_izquierda')
        ]

    puntos_k = transformar_kinovea(puntos_k)

    for marcador, (x, y) in puntos_k.items():
        #if marcador in ["cadera_izquierda", "rodilla_izquierda", "tobillo_izquierdo", "cadera_derecha"]:
        ax.scatter(x, y, color='blue')
    for a, b in conexiones:
        if a in puntos_k and b in puntos_k:
            x1, y1 = puntos_k[a]
            x2, y2 = puntos_k[b]
            ax.plot([x1, x2], [y1, y2], color='blue', linestyle='-', linewidth=1)

    for marcador, (x, y) in puntos_m.items():
        if not (marcador == "hombro_derecho" and lado == "derecha"):
        #if marcador in ["cadera_izquierda", "rodilla_izquierda", "tobillo_izquierdo", "cadera_derecha"]:
            ax.scatter(x, y, color='red')
    for a, b in conexiones:
        if a in puntos_m and b in puntos_m:
            x1, y1 = puntos_m[a]
            x2, y2 = puntos_m[b]
            ax.plot([x1, x2], [y1, y2], color='red', linestyle='--', linewidth=1)

    if lado == "izquierda":
        ax.set_xlim(200, 1200)
    else: 
        ax.set_xlim(700, 1200)

    ax.set_ylim(600, 900)
    ax.invert_yaxis()
    ax.set_title(f"Comparación XY - Archivo {archivo} - Frame {frame}\nAzul: Kinovea | Rojo: MediaPipe")
    ax.set_xlabel("X (px)")
    ax.set_ylabel("Y (px)")
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True)
    plt.tight_layout()
    plt.pause(0.5)
    plt.close()

    carpeta_salida = "plots_kinovea_mediapipe"
    os.makedirs(carpeta_salida, exist_ok=True)

    ruta_salida = os.path.join(carpeta_salida, f"{archivo}_frame{frame}_{lado}.png")
    fig.canvas.draw()
    fig.savefig(ruta_salida, dpi=300)
    plt.close(fig)
    print(f"Gráfico guardado en: {ruta_salida}")


#Función que, dados los archivos que contienen los datos de los puntos de mediapipe y de Kinovea, 
#Devuelve un dataframe con el calculo de los angulos para cada herramienta
def comparar_angulos(path_kinovea, path_mediapipe):
    df_kinovea = cargar_datos(path_kinovea)
    df_mediapipe = cargar_datos(path_mediapipe)

    datos_kinovea = construir_diccionario(df_kinovea)
    datos_mediapipe = construir_diccionario(df_mediapipe)

    # Detectar si los datos son para pierna izquierda o derecha según el nombre del archivo
    if "LI" in path_kinovea.upper():
        lado = "izquierda"
        caderaD = "cadera_izquierda"
        rodillaD = "rodilla_izquierda"
        tobilloD = "tobillo_izquierdo"
        talonD = "talon_izquierdo"
        puntaD = "punta_izquierda"
    elif 'LD' in path_kinovea.upper():
        lado = "derecha"
        caderaD = "cadera_derecha"
        rodillaD = "rodilla_derecha"
        tobilloD = "tobillo_derecho"
        talonD = "talon_derecho"
        puntaD = "punta_derecha"
    else: 
        lado = "trasera"
        caderaD = "cadera_izquierda"
        rodillaD = "rodilla_izquierda"
        tobilloD = "tobillo_izquierdo"
    resultados = []

    claves_comunes = set(datos_kinovea.keys()) & set(datos_mediapipe.keys())
    for archivo_frame in sorted(claves_comunes):
        archivo, frame = archivo_frame
        print("Archivo analizado ",archivo)

        puntos_k = datos_kinovea[archivo_frame]
        puntos_m = datos_mediapipe[archivo_frame]

        try:
            angulo_rodilla_k = calcular_angulo(puntos_k[caderaD], puntos_k[rodillaD], puntos_k[tobilloD])
            angulo_pie_k = calcular_angulo(puntos_k[tobilloD], puntos_k[talonD], puntos_k[puntaD])
        except Exception as e:
            angulo_rodilla_k = None
            angulo_pie_k = None

        try:
            angulo_rodilla_m = calcular_angulo(puntos_m[caderaD], puntos_m[rodillaD], puntos_m[tobilloD])
            angulo_pie_m = calcular_angulo(puntos_m[tobilloD], puntos_m[talonD], puntos_m[puntaD])
        except Exception as e:
            angulo_rodilla_m = None
            angulo_pie_m = None

        if archivo == 42:
            graficar_comparacion(puntos_k, puntos_m, archivo, frame,lado)

        if lado != "trasera":
            resultados.append({
                'archivo': archivo,
                'frame': frame,
                'angulo_rodilla_kinovea': angulo_rodilla_k,
                'angulo_rodilla_mediapipe': angulo_rodilla_m,
                'angulo_pie_kinovea': angulo_pie_k,
                'angulo_pie_mediapipe': angulo_pie_m
            })
        else: 
            resultados.append({
                'archivo': archivo,
                'frame': frame,
                'angulo_rodilla_izquierda_kinovea': angulo_rodilla_k,
                'angulo_rodilla_izquierda_mediapipe': angulo_rodilla_m,
            })


    return pd.DataFrame(resultados)



path_kino = 'G:/Mi unidad/Videos Trabajo de Grado/Excel Kinovea LI/excel_unido.csv'
path_mediapipe = "G:/Mi unidad/Videos Trabajo de Grado/puntos_clave_LI_final.csv"
lado = "T"

df_angulos = comparar_angulos(path_kino, path_mediapipe)

antes = len(df_angulos)
df_angulos = df_angulos.dropna()
despues = len(df_angulos)
print("Antes: ", antes, " Despues: ", despues)

if (lado == "T"):
    df_angulos['diferencia_rodilla_izquierda'] = round(df_angulos['angulo_rodilla_izquierda_kinovea'] - df_angulos['angulo_rodilla_izquierda_mediapipe'],3)
else:
    df_angulos['diferencia_rodilla'] = round(df_angulos['angulo_rodilla_kinovea'] - df_angulos['angulo_rodilla_mediapipe'],3)
    df_angulos['diferencia_pie'] = round(df_angulos['angulo_pie_kinovea'] - df_angulos['angulo_pie_mediapipe'],3)
#print(df_angulos)

df_angulos.to_csv("comparacion_LI_final.csv", index=False)


