from django.shortcuts import render, HttpResponse
from django.core.files.storage import FileSystemStorage
import cv2
import os

def rotar_video(input_path, output_path):
    path = 'C:/Users/laura/OneDrive/Documents/TrabajoGrado_LauraSalamanca/Codigo/interfaz/' + input_path
    path_final = 'C:/Users/laura/OneDrive/Documents/TrabajoGrado_LauraSalamanca/Codigo/interfaz/' + output_path
    print(path, " --- ",path_final)
    captura = cv2.VideoCapture(path)
    
    if not captura.isOpened():
        return

    # Obtener las propiedades del video
    fps = int(captura.get(cv2.CAP_PROP_FPS))  
    width = int(captura.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(captura.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = int(captura.get(cv2.CAP_PROP_FOURCC))  
    total_frames = int(captura.get(cv2.CAP_PROP_FRAME_COUNT))  

    writer = cv2.VideoWriter(path_final, fourcc, fps, (width, height))

    while captura.isOpened():
        ret, frame = captura.read()
        if not ret:
            break
        
        # Aplicar rotación de 180 grados
        frame = cv2.rotate(frame, cv2.ROTATE_180)  # Rotar el frame 180 grados
        
        # Escribir el frame procesado al archivo de salida
        writer.write(frame)

    # Liberar los recursos
    captura.release()
    writer.release()

def index(request):
    video_path = None
    if request.method == "POST" and 'video' in request.FILES:
        # Cargar el video desde el formulario
        video = request.FILES['video']
        
        # Crear una instancia de FileSystemStorage para guardar el archivo
        fs = FileSystemStorage()
        filename = fs.save(video.name, video)
        video_path = fs.url(filename)  # La URL del video original

        # Generar la ruta de salida para el video rotado en formato .mp4
        output_path = os.path.join('media', 'rotated_' + filename)
        output_path = output_path.replace('\\', '/')
        # Llamar a la función para rotar el video y guardarlo como .mp4
        rotar_video(video_path[1:], output_path)  # Usamos [1:] para eliminar el primer caracter '/' 

        # Retornar la respuesta con el video rotado
        print("/"+output_path)
        print(video_path)
        return render(request, 'index.html', {
            'video_path': ("/"+output_path)  # Pasa la ruta del video procesado
        })

    return render(request, 'index.html', {'video_path': video_path})
