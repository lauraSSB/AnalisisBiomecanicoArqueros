import os

path = "G:/Mi unidad/Videos Trabajo Grado/Trasera (Andy)"

archivos = sorted(os.listdir(path))

for i, nombre_archivo in enumerate(archivos, start=1):
    path_ant = os.path.join(path, nombre_archivo)
    
    file_extension = os.path.splitext(nombre_archivo)[1]
    
    nombre_nuevo = f"Piso_LI_{i}{file_extension}"
    path_nuevo = os.path.join(path, new_name)

    try:
        os.rename(path_ant, path_nuevo)
        print(f"Renombrado: {nombre_archivo} -> {nombre_nuevo}")
    except Exception as e:
        print(f"Error al renombrar: {nombre_archivo} (Índice: {i})")

print("Renombrado completado (con errores indicados si los hubo).")
