from django.shortcuts import render, HttpResponse
from django.core.files.storage import FileSystemStorage
import cv2
import os

def rotar_video(input_path, output_path):
    path = 'C:/Users/laura/OneDrive/Documents/TrabajoGrado_LauraSalamanca/Codigo/interfaz/' + input_path
    path_final = 'C:/Users/laura/OneDrive/Documents/TrabajoGrado_LauraSalamanca/Codigo/interfaz/' + output_path
    print(path, " --- ",path_final)

    captura = cv2.VideoCapture(path,cv2.CAP_FFMPEG)
    
    if not captura.isOpened():
        return

    fps = int(captura.get(cv2.CAP_PROP_FPS))  
    width = int(captura.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(captura.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = int(captura.get(cv2.CAP_PROP_FOURCC))  
    total_frames = int(captura.get(cv2.CAP_PROP_FRAME_COUNT))  

    writer = cv2.VideoWriter(path_final, fourcc, fps, (width, height))
    if not writer.isOpened():
        print(f"Error al abrir el VideoWriter para {output_path}")
        return
    while captura.isOpened():
        ret, frame = captura.read()
               
        if not ret:
            break
        frame = cv2.rotate(frame, cv2.ROTATE_180)  # Rota la imagen 180°
        cv2.imshow("Frame",frame)
        writer.write(frame)

    captura.release()
    cv2.destroyAllWindows()
    writer.release()
    

# def index(request):
#     video_path = None
#     if request.method == "POST" and 'video' in request.FILES:
#         video = request.FILES['video']
#         fs = FileSystemStorage()
#         filename = fs.save(video.name, video)
#         video_path = fs.url(filename)  

#         output_path = os.path.join('media', 'rotated_' + filename)
#         output_path = output_path.replace('\\', '/')

#         # Llamar a la función para rotar el video y guardarlo como .mp4
#         rotar_video(video_path[1:], output_path)  # Usamos [1:] para eliminar el primer caracter '/' 

#         # Retornar la respuesta con el video rotado
#         print("/"+output_path)
#         print(video_path)
#         return render(request, 'index.html', {
#             'video_path': (video_path)  # Pasa la ruta del video procesado
#         })

#     return render(request, 'index.html', {'video_path': video_path})


def index(request):
    # Cuando se recibe el formulario
    if request.method == "POST" and request.FILES:
        # Obtener los videos de la solicitud
        video1 = request.FILES.get('video1')
        video2 = request.FILES.get('video2')
        video3 = request.FILES.get('video3')

        # Crear un FileSystemStorage para guardar los archivos
        fs = FileSystemStorage()

        # Guardar los archivos en el directorio adecuado y obtener sus URLs
        video1_path = fs.save(video1.name, video1)
        video2_path = fs.save(video2.name, video2)
        video3_path = fs.save(video3.name, video3)

        # Obtener las URLs de los videos guardados
        video1_url = fs.url(video1_path)
        video2_url = fs.url(video2_path)
        video3_url = fs.url(video3_path)

        print()
        # Pasar las URLs de los videos al template para mostrarlos
        return render(request, 'index.html', {
            'video1_url': video1_url,
            'video2_url': video2_url,
            'video3_url': video3_url
        })

    return render(request, 'index.html')