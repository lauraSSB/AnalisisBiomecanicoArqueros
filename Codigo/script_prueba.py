import cv2
import numpy as np

# Abrir el video
captura = cv2.VideoCapture('videoprueba.mp4')

if not captura.isOpened():
    print("Error al abrir el video")
    exit()

# Reducción del tamaño al 80%
scale_factor = 0.8

# Rango de color del balón (ajustar según el color del balón)
lower_color = np.array([0, 100, 100])   # Rojo/Naranja
upper_color = np.array([30, 255, 255])  # Amarillo/Rojo fuerte

while True:
    ret, frame = captura.read()
    if not ret:
        print("Fin del video o error al leer el frame")
        break

    # Redimensionar el frame al 80%
    frame = cv2.resize(frame, (int(frame.shape[1] * scale_factor), int(frame.shape[0] * scale_factor)))

    # Convertir a HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Crear una máscara para el balón
    mask = cv2.inRange(hsv, lower_color, upper_color)

    # Aplicar morfología para reducir ruido
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))

    # Encontrar contornos en la máscara
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        
        if area > 500:  # Filtrar objetos muy pequeños
            # Obtener el rectángulo delimitador
            x, y, w, h = cv2.boundingRect(cnt)

            # Dibujar rectángulo alrededor del balón
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Calcular centro del balón
            cx, cy = x + w // 2, y + h // 2

            # Mostrar coordenadas en pantalla
            cv2.putText(frame, f"Balón: ({cx}, {cy})", (cx - 40, cy - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Mostrar el video con detección de balón
    cv2.imshow("Detección de Balón", frame)

    # Presiona 'q' para salir
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Liberar recursos
captura.release()
cv2.destroyAllWindows()
