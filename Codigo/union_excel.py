import pandas as pd
import os

carpeta = 'G:/Mi unidad/Videos Trabajo de Grado/Excel Kinovea LD'

if 'LD' in carpeta:
    partes = ['punta_derecha', 'talon_derecho', 'tobillo_derecho', 'rodilla_derecha', 'cadera_derecha']
elif 'LI' in carpeta:
    partes = ['punta_izquierda', 'talon_izquierdo', 'tobillo_izquierdo', 'rodilla_izquierda', 'cadera_izquierda']


registros = []

for archivo in os.listdir(carpeta):
    if archivo.endswith('.xlsx'):
        ruta = os.path.join(carpeta, archivo)
        numero_archivo = ''.join(filter(str.isdigit, archivo))  
        print(ruta)

        if not archivo[0].isalnum(): #no se porque me aparecen archivos raros pero no se deben contar
            continue

        df = pd.read_excel(ruta, header=None)
        df = df.dropna(axis=0, how='all')
        df = df.dropna(axis=1, how='all')
        df = df.reset_index(drop=True)

        indices = {}
        for parte in partes:
            apariciones = df[df[0] == parte].index.tolist()
            if len(apariciones) >= 2:
                indices[parte] = apariciones[1]
            else: #por errores de escritura a veces esta mal
                print(f"Revisar '{parte}' en el archivo: '{archivo}'")
        
        ordenadas = sorted(indices.items(), key=lambda x: x[1])

        for i, (parte, inicio) in enumerate(ordenadas):
            fila_inicio = inicio + 3
            fila_fin = ordenadas[i+1][1] if i+1 < len(ordenadas) else len(df)
            
            datos = df.iloc[fila_inicio:fila_fin].copy()
            datos = datos.dropna(how='all')
            datos.columns = ['Time (s)', 'X (px)', 'Y (px)', 'extra'][:len(datos.columns)]
            datos = datos[['X (px)', 'Y (px)']]
            datos = datos.reset_index(drop=True)

            for frame, fila in datos.iterrows():
                registros.append({
                    'archivo': int(numero_archivo),
                    'marcador': parte,
                    'frame': frame,
                    'X (px)': fila['X (px)'],
                    'Y (px)': fila['Y (px)']
                })

df_final = pd.DataFrame(registros)

print(df_final)

df_final.to_csv('G:/Mi unidad/Videos Trabajo de Grado/Excel Kinovea LD/excel_unido.csv', index=False)
