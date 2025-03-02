import cv2
import os

video = r"C:\Users\laura\OneDrive - Pontificia Universidad Javeriana\Videos Tesis\Saques de Piso\LateralDerecha (Lau)\Piso_LD_5.MOV"

carpeta_frame = r"C:\Users\laura\OneDrive\Documents\TrabajoGrado_LauraSalamanca\Frames"

os.makedirs(carpeta_frame, exist_ok=True)

output_path = os.path.join(carpeta_frame, "frame2.jpg")

captura = cv2.VideoCapture(video)

if not captura.isOpened():
    print("Error al abrir el video.")
    exit()

num_frame = 190  # Número del frame que quieres extraer

captura.set(cv2.CAP_PROP_POS_FRAMES, num_frame)

ret, frame = captura.read()

if ret:
    cv2.imshow("Frame Extraído", frame)

    cv2.imwrite(output_path, frame)
    print(f"Frame guardado en: {carpeta_frame}")

    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("No se pudo leer el frame.")
    
captura.release()
