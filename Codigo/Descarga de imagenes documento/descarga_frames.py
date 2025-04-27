import cv2
import os
import mediapipe as mp
import numpy as np
import math


def calcular_angulo(p1, p2, p3):
    a = np.array([p1.x, p1.y, p1.z])
    b = np.array([p2.x, p2.y, p2.z])
    c = np.array([p3.x, p3.y, p3.z])

    ba = a - b
    bc = c - b

    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    return round(np.degrees(angle),2)

#Función que descarga un 5 frames siguientes a un determinado frame de un video. Se puede especificar la rita de descarga
def descarga_normal(video, carpeta_frame, num_frame):
    os.makedirs(carpeta_frame, exist_ok=True)

    output_path = os.path.join(carpeta_frame, "frame3.jpg")

    captura = cv2.VideoCapture(video)

    if not captura.isOpened():
        print("Error al abrir el video.")
        exit()

    for i in range(0,1):
        print(num_frame+i)
        captura.set(cv2.CAP_PROP_POS_FRAMES, num_frame+i)

        ret, frame = captura.read()

        frame = cv2.rotate(frame, cv2.ROTATE_180)

        if ret:
            cv2.imshow("Frame Extraído", frame)

            cv2.imwrite(output_path, frame)
            print(f"Frame guardado en: {carpeta_frame}")

            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("No se pudo leer el frame.")
        
    captura.release()

def dibujar_piernas(frame, landmarks, mp_pose):
    PIERNA_DERECHA = [
        (mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE),
        (mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.RIGHT_ANKLE),
        (mp_pose.PoseLandmark.RIGHT_ANKLE, mp_pose.PoseLandmark.RIGHT_HEEL),
        (mp_pose.PoseLandmark.RIGHT_ANKLE, mp_pose.PoseLandmark.RIGHT_FOOT_INDEX),
        (mp_pose.PoseLandmark.RIGHT_HEEL, mp_pose.PoseLandmark.RIGHT_FOOT_INDEX),
    ]

    PIERNA_IZQUIERDA = [
        (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE),
        (mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.LEFT_ANKLE),
        (mp_pose.PoseLandmark.LEFT_ANKLE, mp_pose.PoseLandmark.LEFT_HEEL),
        (mp_pose.PoseLandmark.LEFT_ANKLE, mp_pose.PoseLandmark.LEFT_FOOT_INDEX),
        (mp_pose.PoseLandmark.LEFT_HEEL, mp_pose.PoseLandmark.LEFT_FOOT_INDEX),
    ]

    h, w, _ = frame.shape

    print(mp_pose.PoseLandmark.RIGHT_KNEE," - ",mp_pose.PoseLandmark.RIGHT_ANKLE)
    if (landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y) > (landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y):
        b = 255
        r = 0
    else:
        b = 0
        r = 255

    for connection in PIERNA_DERECHA:
        p1 = landmarks[connection[0].value]
        p2 = landmarks[connection[1].value]
        cv2.line(frame, (int(p1.x * w), int(p1.y * h)), (int(p2.x * w), int(p2.y * h)), (b, 0, r), 2)
        cv2.circle(frame, (int(p1.x * w), int(p1.y * h)), 5, (b, 0, r), -1)
        cv2.circle(frame, (int(p2.x * w), int(p2.y * h)), 5, (b, 0, r), -1)

    # for connection in PIERNA_IZQUIERDA:
    #     p1 = landmarks[connection[0].value]
    #     p2 = landmarks[connection[1].value]
    #     cv2.line(frame, (int(p1.x * w), int(p1.y * h)), (int(p2.x * w), int(p2.y * h)), (0, 255, 0), 2)
    #     cv2.circle(frame, (int(p1.x * w), int(p1.y * h)), 5, (0, 255, 0), -1)
    #     cv2.circle(frame, (int(p2.x * w), int(p2.y * h)), 5, (0, 255, 0), -1)

def descarga_piernas(video, carpeta_frame, num_frame):
    os.makedirs(carpeta_frame, exist_ok=True)
    
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.8, min_tracking_confidence=0.80, model_complexity=2)
    
    captura = cv2.VideoCapture(video)
    
    if not captura.isOpened():
        print("Error al abrir el video.")
        exit()
    
    for i in range(-1, 4):
        captura.set(cv2.CAP_PROP_POS_FRAMES, num_frame + i)
        ret, frame = captura.read()
        if not ret:
            print("No se pudo leer el frame.")
            continue
        
        frame = cv2.resize(frame, (int(frame.shape[1] * 0.6), int(frame.shape[0] * 0.6)))
        frame = cv2.rotate(frame, cv2.ROTATE_180)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resultado = pose.process(rgb_frame)
        
        if resultado.pose_landmarks:
            dibujar_piernas(frame, resultado.pose_landmarks.landmark, mp_pose)
        
        output_path = os.path.join(carpeta_frame, f"LI_inicio{i}.jpg")
        cv2.imwrite(output_path, frame)
        #cv2.imshow("Frame",frame)
        print(f"Frame guardado en: {output_path}")
    
    captura.release()

def analizar_rodilla_tobillo(video, num_frame, carpeta_frame):
    os.makedirs(carpeta_frame, exist_ok=True)
    
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.80, min_tracking_confidence=0.90, model_complexity=2)
    
    captura = cv2.VideoCapture(video)
    
    if not captura.isOpened():
        print("Error al abrir el video.")
        exit()

    captura.set(cv2.CAP_PROP_POS_FRAMES, num_frame)
    ret, frame = captura.read()
    if not ret:
        print("No se pudo leer el frame.")
        captura.release()
        return
    
    # Redimensionar el frame para visualización
    #frame = cv2.rotate(frame, cv2.ROTATE_180)
    #rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    resultado = pose.process(frame)
    
    if resultado.pose_landmarks:
        # Coordenadas de rodilla y tobillo derecho
        rodilla_derecha = resultado.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE]
        tobillo_derecha = resultado.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE]

        h, w, _ = frame.shape

        # Dibujar puntos de rodilla y tobillo en el frame
        cv2.circle(frame, (int(rodilla_derecha.x * w), int(rodilla_derecha.y * h)), 5, (255, 255, 0), -1)  # Rodilla
        cv2.circle(frame, (int(tobillo_derecha.x * w), int(tobillo_derecha.y * h)), 5, (0, 0, 255), -1)  # Tobillo
        
        # Mostrar coordenadas en Y

        cv2.putText(frame, f"Rodilla Y: {rodilla_derecha.y * h:.2f}", (int(rodilla_derecha.x * w) + 10, int(rodilla_derecha.y * h) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), 4)
        cv2.putText(frame, f"Tobillo Y: {tobillo_derecha.y * h:.2f}", (int(tobillo_derecha.x * w) + 10, int(tobillo_derecha.y * h) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)

        diferencia_y = rodilla_derecha.y*h - tobillo_derecha.y*h

        cv2.putText(frame, f"Distancia en Y: {diferencia_y:.2f}", (frame.shape[1] - 700, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 4)
    
    # Guardar el frame procesado
    output_path = os.path.join(carpeta_frame, f"Frame_{num_frame}_rodilla.jpg")
    cv2.imwrite(output_path, frame)
    print(f"Frame guardado en: {output_path}")
    
    # Mostrar el frame
    cv2.imshow("Frame con Rodilla y Tobillo", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    captura.release()

def analizar_piernas_adelante(video, num_frame, carpeta_frame):
    os.makedirs(carpeta_frame, exist_ok=True)
    
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.80, min_tracking_confidence=0.90, model_complexity=2)
    
    captura = cv2.VideoCapture(video)
    
    if not captura.isOpened():
        print("Error al abrir el video.")
        exit()

    captura.set(cv2.CAP_PROP_POS_FRAMES, num_frame)
    ret, frame = captura.read()
    frame = cv2.rotate(frame, cv2.ROTATE_180)
    if not ret:
        print("No se pudo leer el frame.")
        captura.release()
        return
    
    resultado = pose.process(frame)
    
    if resultado.pose_landmarks:
        h, w, _ = frame.shape
        landmark = resultado.pose_landmarks.landmark

        rodilla_d = landmark[mp_pose.PoseLandmark.RIGHT_KNEE]
        tobillo_d = landmark[mp_pose.PoseLandmark.RIGHT_ANKLE]

        # Rodilla y tobillo izquierdo
        rodilla_i = landmark[mp_pose.PoseLandmark.LEFT_KNEE]
        tobillo_i = landmark[mp_pose.PoseLandmark.LEFT_ANKLE]

        x_rod_d = int(rodilla_d.x * w)
        x_tob_d = int(tobillo_d.x * w)
        x_rod_i = int(rodilla_i.x * w)
        x_tob_i = int(tobillo_i.x * w)

        y_rod_d = int(rodilla_d.y * h)
        y_tob_d = int(tobillo_d.y * h)
        y_rod_i = int(rodilla_i.y * h)
        y_tob_i = int(tobillo_i.y * h)

        # Dibujo de puntos
        #cv2.circle(frame, (x_rod_d, y_rod_d), 6, (0, 0, 255), -1)  # Rodilla derecha
        cv2.circle(frame, (x_tob_d, y_tob_d), 6, (0, 0, 255), -1)    # Tobillo derecho

        #cv2.circle(frame, (x_rod_i, y_rod_i), 6, (255, 255, 0), -1)  # Rodilla izquierda
        cv2.circle(frame, (x_tob_i, y_tob_i), 6, (255, 0, 0), -1)    # Tobillo izquierdo

        
        #cv2.putText(frame, f"Rodilla Derecha X: {rodilla_d.x * w:.2f}", (x_rod_d - 400, y_rod_d - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.7, (0,0, 255), 3)
        cv2.putText(frame, f"Tobillo Derecho X: {tobillo_d.x * w:.2f}", (x_tob_d - 400, y_tob_d - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.7, (0, 0, 255), 3)
        
        #cv2.putText(frame, f"Rodilla Izquierda X: {rodilla_i.x * w:.2f}", (x_rod_i - 500, y_rod_i +50), cv2.FONT_HERSHEY_SIMPLEX, 1.7, (255, 255,0), 3)
        cv2.putText(frame, f"Tobillo Izquierdo X: {tobillo_i.x * w:.2f}", (x_tob_i - 500, y_tob_i + 50), cv2.FONT_HERSHEY_SIMPLEX, 1.7, (255, 0, 0), 3)

        dif_x_d = rodilla_d.x * w - tobillo_d.x * w
        dif_x_i = rodilla_i.x * w - tobillo_i.x * w

        # cv2.putText(frame, f"Dist X D: {dif_x_d:.2f}", (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255, 0, 0), 4)
        # cv2.putText(frame, f"Dist X I: {dif_x_i:.2f}", (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 0, 255), 4)

    output_path = os.path.join(carpeta_frame, f"Frame_{num_frame}_piernas_adelante_tobillo_LD.jpg")
    cv2.imwrite(output_path, frame)
    print(f"Frame guardado en: {output_path}")

    cv2.imshow("Frame con Piernas", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    captura.release()

def analizar_tobillos_z(video, num_frame, carpeta_frame):
    os.makedirs(carpeta_frame, exist_ok=True)
    
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.80, min_tracking_confidence=0.90, model_complexity=2)
    
    captura = cv2.VideoCapture(video)
    
    if not captura.isOpened():
        print("Error al abrir el video.")
        exit()

    captura.set(cv2.CAP_PROP_POS_FRAMES, num_frame)
    ret, frame = captura.read()
    frame = cv2.rotate(frame, cv2.ROTATE_180)
    if not ret:
        print("No se pudo leer el frame.")
        captura.release()
        return
    
    resultado = pose.process(frame)
    
    if resultado.pose_landmarks:
        h, w, _ = frame.shape
        landmark = resultado.pose_landmarks.landmark

        tobillo_d = landmark[mp_pose.PoseLandmark.RIGHT_KNEE]
        tobillo_i = landmark[mp_pose.PoseLandmark.LEFT_KNEE]

        x_tob_d = int(tobillo_d.x * w)
        y_tob_d = int(tobillo_d.y * h)

        x_tob_i = int(tobillo_i.x * w)
        y_tob_i = int(tobillo_i.y * h)

        # Dibujo de tobillos
        cv2.circle(frame, (x_tob_d, y_tob_d), 6, (0, 0, 255), -1)    # Tobillo derecho
        cv2.circle(frame, (x_tob_i, y_tob_i), 6, (0, 255, 0), -1)    # Tobillo izquierdo

        # Mostrar coordenadas Z
        cv2.putText(frame, f"Rodilla Derecha: {tobillo_d.z:.4f}", (x_tob_d + 10, y_tob_d - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.7, (0, 0, 255), 3)
        cv2.putText(frame, f"Rodilla Izquierda: {tobillo_i.z:.4f}", (x_tob_i + 10, y_tob_i + 30), cv2.FONT_HERSHEY_SIMPLEX, 1.7, (0, 255, 0), 3)

    output_path = os.path.join(carpeta_frame, f"Frame_{num_frame}_rodillas_z.jpg")
    cv2.imwrite(output_path, frame)
    print(f"Frame guardado en: {output_path}")

    cv2.imshow("Coordenadas Z de tobillos", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    captura.release()

def dibujar_pierna_derecha(video, num_frame, carpeta_frame):
    import time
    os.makedirs(carpeta_frame, exist_ok=True)

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.70, min_tracking_confidence=0.90, model_complexity=2)

    captura = cv2.VideoCapture(video)

    if not captura.isOpened():
        print("Error al abrir el video.")
        return

    frames_a_tomar = [num_frame, num_frame + 1,num_frame + 2]

    for i, frame_id in enumerate(frames_a_tomar):
        captura.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = captura.read()
        frame = cv2.rotate(frame, cv2.ROTATE_180)
        if not ret:
            print(f"No se pudo leer el frame {frame_id}")
            continue

        resultado = pose.process(frame)

        if resultado.pose_landmarks:
            h, w, _ = frame.shape
            landmark = resultado.pose_landmarks.landmark

            # Puntos clave de la pierna de pateo (derecha por defecto)
            cadera = landmark[mp_pose.PoseLandmark.RIGHT_HIP]
            rodilla = landmark[mp_pose.PoseLandmark.RIGHT_KNEE]
            tobillo = landmark[mp_pose.PoseLandmark.RIGHT_ANKLE]

            # Coordenadas absolutas
            x_cadera, y_cadera = int(cadera.x * w), int(cadera.y * h)
            x_rodilla, y_rodilla = int(rodilla.x * w), int(rodilla.y * h)
            x_tobillo, y_tobillo = int(tobillo.x * w), int(tobillo.y * h)

            # Dibujar puntos
            cv2.circle(frame, (x_cadera, y_cadera), 8, (255, 255, 0), -1)
            cv2.circle(frame, (x_rodilla, y_rodilla), 8, (0, 255, 0), -1)
            cv2.circle(frame, (x_tobillo, y_tobillo), 8, (0, 0, 255), -1)

            # Dibujar líneas entre puntos
            cv2.line(frame, (x_cadera, y_cadera), (x_rodilla, y_rodilla), (255, 255, 255), 2)
            cv2.line(frame, (x_rodilla, y_rodilla), (x_tobillo, y_tobillo), (0, 255, 255), 2)

        else:
            print("No reconocio pose")
        # Guardar frame
        output_path = os.path.join(carpeta_frame, f"Frame_{i}_pierna_pateo_T.jpg")
        cv2.imwrite(output_path, frame)
        print(f"Frame guardado en: {output_path}")

        # Mostrar frame en pantalla
        cv2.imshow(f"Frame {frame_id} - Pierna de pateo", frame)
        
        #time.sleep(5)
        cv2.destroyAllWindows()

    captura.release()

def analizar_angulos_derecha(video, num_frame, carpeta_frame):
    os.makedirs(carpeta_frame, exist_ok=True)
    
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.80, min_tracking_confidence=0.90, model_complexity=2)
    
    captura = cv2.VideoCapture(video)
    
    if not captura.isOpened():
        print("Error al abrir el video.")
        exit()

    captura.set(cv2.CAP_PROP_POS_FRAMES, num_frame)
    ret, frame = captura.read()
    #frame = cv2.rotate(frame, cv2.ROTATE_180)
    if not ret:
        print("No se pudo leer el frame.")
        captura.release()
        return
    
    resultado = pose.process(frame)
    
    if resultado.pose_landmarks:
        h, w, _ = frame.shape
        landmark = resultado.pose_landmarks.landmark

        hombro_d = landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        cadera_d = landmark[mp_pose.PoseLandmark.RIGHT_HIP]
        rodilla_d = landmark[mp_pose.PoseLandmark.RIGHT_KNEE]
        tobillo_d = landmark[mp_pose.PoseLandmark.RIGHT_ANKLE]
        talon_d = landmark[mp_pose.PoseLandmark.RIGHT_HEEL]
        punta_d = landmark[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX]

        # Cálculo de ángulos
        angulo_cadera = calcular_angulo(rodilla_d,cadera_d, hombro_d )
        angulo_rodilla = calcular_angulo(cadera_d, rodilla_d, tobillo_d)
        angulo_pie = calcular_angulo(tobillo_d, talon_d, punta_d)

        # Dibujar puntos
        for punto in [hombro_d, cadera_d, rodilla_d, tobillo_d, talon_d, punta_d]:
            cv2.circle(frame, (int(punto.x * w), int(punto.y * h)), 5, (0, 255, 255), -1)

        # Dibujar líneas para los ángulos
        cv2.line(frame, (int(hombro_d.x * w), int(hombro_d.y * h)), (int(cadera_d.x * w), int(cadera_d.y * h)), (0, 255, 0), 1)
        cv2.line(frame, (int(cadera_d.x * w), int(cadera_d.y * h)), (int(rodilla_d.x * w), int(rodilla_d.y * h)), (0, 255, 0), 1)

        cv2.line(frame, (int(cadera_d.x * w), int(cadera_d.y * h)), (int(rodilla_d.x * w), int(rodilla_d.y * h)), (0, 255, 0), 1)
        cv2.line(frame, (int(rodilla_d.x * w), int(rodilla_d.y * h)), (int(tobillo_d.x * w), int(tobillo_d.y * h)), (0, 255, 0), 1)

        cv2.line(frame, (int(tobillo_d.x * w), int(tobillo_d.y * h)), (int(talon_d.x * w), int(talon_d.y * h)), (0, 255, 0), 1)
        cv2.line(frame, (int(talon_d.x * w), int(talon_d.y * h)), (int(punta_d.x * w), int(punta_d.y * h)), (0, 255, 0), 1)

        # Dibujar textos con los ángulos
        cv2.putText(frame, f"{angulo_cadera:.2f}", (int(cadera_d.x * w) + 40, int(cadera_d.y * h) + 10 ),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)

        cv2.putText(frame, f"{angulo_rodilla:.2f}", (int(rodilla_d.x * w) + 20, int(rodilla_d.y * h) + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)

        cv2.putText(frame, f"{angulo_pie:.2f}", (int(talon_d.x * w) - 50, int(tobillo_d.y * h) - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)

    output_path = os.path.join(carpeta_frame, f"Frame_{num_frame}_angulos_LD.jpg")
    cv2.imwrite(output_path, frame)
    print(f"Frame guardado en: {output_path}")

    cv2.imshow("Frame con Piernas", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    captura.release()

def calcular_angulo_vertical(hombro, cadera):
    v = np.array([hombro.x - cadera.x, hombro.y - cadera.y])  # Vector desde cadera hacia hombro
    vertical = np.array([0, -1])  # Eje Y hacia arriba

    cos_theta = np.dot(v, vertical) / (np.linalg.norm(v) * np.linalg.norm(vertical))
    angulo = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))

    # Signo: si el hombro está hacia delante (más a la derecha que la cadera), es positivo
    signo = np.sign(hombro.x - cadera.x)

    return signo * angulo

def analizar_angulos_izquierda(video, num_frame, carpeta_frame):
    os.makedirs(carpeta_frame, exist_ok=True)

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.80, min_tracking_confidence=0.90, model_complexity=2)

    captura = cv2.VideoCapture(video)

    if not captura.isOpened():
        print("Error al abrir el video.")
        exit()

    captura.set(cv2.CAP_PROP_POS_FRAMES, num_frame)
    ret, frame = captura.read()
    frame = cv2.rotate(frame, cv2.ROTATE_180)

    if not ret:
        print("No se pudo leer el frame.")
        captura.release()
        return

    resultado = pose.process(frame)

    if resultado.pose_landmarks:
        h, w, _ = frame.shape
        landmark = resultado.pose_landmarks.landmark

        hombro_i = landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        cadera_i = landmark[mp_pose.PoseLandmark.LEFT_HIP]
        rodilla_i = landmark[mp_pose.PoseLandmark.LEFT_KNEE]
        tobillo_i = landmark[mp_pose.PoseLandmark.LEFT_ANKLE]
        talon_i = landmark[mp_pose.PoseLandmark.LEFT_HEEL]
        punta_i = landmark[mp_pose.PoseLandmark.LEFT_FOOT_INDEX]

        # Línea punteada vertical desde cadera izquierda (eje Y)
        x_cadera = int(cadera_i.x * w)
        y_inicio = 0
        y_fin = h
        punteado_largo = 10
        espacio = 10

        for y in range(y_inicio, y_fin, punteado_largo + espacio):
            cv2.line(frame, (x_cadera, y), (x_cadera, min(y + punteado_largo, y_fin)), (200, 200, 200), 2)


        # Cálculo de ángulos
        angulo_inclinacion = calcular_angulo_vertical(hombro_i, cadera_i)
        angulo_rodilla = calcular_angulo(cadera_i, rodilla_i, tobillo_i)
        angulo_pie = calcular_angulo(tobillo_i, talon_i, punta_i)

        # Dibujar puntos
        for punto in [hombro_i, cadera_i, rodilla_i, tobillo_i, talon_i, punta_i]:
            cv2.circle(frame, (int(punto.x * w), int(punto.y * h)), 5, (0, 255, 255), -1)

        # Dibujar líneas
        cv2.line(frame, (int(hombro_i.x * w), int(hombro_i.y * h)), (int(cadera_i.x * w), int(cadera_i.y * h)), (255, 0, 0), 2)
        cv2.line(frame, (int(cadera_i.x * w), int(cadera_i.y * h)), (int(rodilla_i.x * w), int(rodilla_i.y * h)), (0, 255, 0), 2)
        cv2.line(frame, (int(rodilla_i.x * w), int(rodilla_i.y * h)), (int(tobillo_i.x * w), int(tobillo_i.y * h)), (0, 255, 0), 2)
        cv2.line(frame, (int(tobillo_i.x * w), int(tobillo_i.y * h)), (int(talon_i.x * w), int(talon_i.y * h)), (0, 255, 0), 2)
        cv2.line(frame, (int(talon_i.x * w), int(talon_i.y * h)), (int(punta_i.x * w), int(punta_i.y * h)), (0, 255, 0), 2)

        # Dibujar textos con los ángulos
        cv2.putText(frame, f"{angulo_inclinacion:.2f}", (int(cadera_i.x * w) + 10, int(cadera_i.y * h)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)

        cv2.putText(frame, f"{angulo_rodilla:.2f}", (int(rodilla_i.x * w) + 20, int(rodilla_i.y * h) + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)

        cv2.putText(frame, f"{angulo_pie:.2f}", (int(talon_i.x * w) + 10, int(tobillo_i.y * h)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)

    output_path = os.path.join(carpeta_frame, f"Frame_{num_frame}_angulos_LI_positivo.jpg")
    cv2.imwrite(output_path, frame)
    print(f"Frame guardado en: {output_path}")

    cv2.imshow("Frame con Pierna Izquierda", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    captura.release()

import numpy as np

def punto_medio_2d(p1, p2):
    return np.array([
        (p1.x + p2.x) / 2,
        (p1.y + p2.y) / 2
    ])

def calcular_angulo_inclinacion_tronco(hombro_izq, hombro_der, cadera_izq, cadera_der, talon):
    # Puntos medios en 2D
    hombros = punto_medio_2d(hombro_izq, hombro_der)
    caderas = punto_medio_2d(cadera_izq, cadera_der)
    talon = np.array([talon.x, talon.y])

    # Vector tronco (cadera -> hombros)
    tronco = hombros - caderas

    # Eje vertical: desde talón hacia caderas
    eje_vertical = caderas - talon

    # Ángulo entre tronco y eje vertical
    cos_theta = np.dot(tronco, eje_vertical) / (np.linalg.norm(tronco) * np.linalg.norm(eje_vertical))
    angulo = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))

    # Signo: positivo si hombros están detrás de caderas (en X), negativo si adelante
    signo = np.sign(hombros[0] - caderas[0])

    return signo * angulo


def analizar_angulos_trasera(video, num_frame, carpeta_frame):
    os.makedirs(carpeta_frame, exist_ok=True)

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.80, min_tracking_confidence=0.90, model_complexity=2)

    captura = cv2.VideoCapture(video)

    if not captura.isOpened():
        print("Error al abrir el video.")
        exit()

    captura.set(cv2.CAP_PROP_POS_FRAMES, num_frame)
    ret, frame = captura.read()
    frame = cv2.rotate(frame, cv2.ROTATE_180)

    if not ret:
        print("No se pudo leer el frame.")
        captura.release()
        return

    resultado = pose.process(frame)

    if resultado.pose_landmarks:
        h, w, _ = frame.shape
        landmark = resultado.pose_landmarks.landmark

        # Landmarks
        hombro_i = landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        hombro_d = landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        cadera_i = landmark[mp_pose.PoseLandmark.LEFT_HIP]
        cadera_d = landmark[mp_pose.PoseLandmark.RIGHT_HIP]
        rodilla_i = landmark[mp_pose.PoseLandmark.LEFT_KNEE]
        tobillo_i = landmark[mp_pose.PoseLandmark.LEFT_ANKLE]
        talon_i = landmark[mp_pose.PoseLandmark.LEFT_HEEL]
        punta_i = landmark[mp_pose.PoseLandmark.LEFT_FOOT_INDEX]

        # Cálculo de ángulos
        angulo_inclinacion = calcular_angulo_inclinacion_tronco(hombro_i, hombro_d, cadera_i, cadera_d, talon_i)
        angulo_rodilla = calcular_angulo(cadera_i, rodilla_i, tobillo_i)
        angulo_pie = calcular_angulo(tobillo_i, talon_i, punta_i)

        # Puntos medios
        hombros = punto_medio_2d(hombro_i, hombro_d)
        caderas = punto_medio_2d(cadera_i, cadera_d)

        # Coordenadas
        px_hombros = (int(hombros[0] * w), int(hombros[1] * h))
        px_caderas = (int(caderas[0] * w), int(caderas[1] * h))
        px_talon = (int(talon_i.x * w), int(talon_i.y * h))

        # Dibujo de líneas
        cv2.line(frame, px_caderas, px_hombros, (255, 0, 0), 2)   # Tronco
        cv2.line(frame, px_talon, px_caderas, (200, 200, 200), 2) # Eje vertical

        # Dibujo de puntos
        for punto in  [talon_i]:
            cv2.circle(frame, (int(punto.x * w), int(punto.y * h)), 5, (0, 255, 255), -1)
        cv2.circle(frame, px_hombros, 5, (0, 255, 255), -1)
        cv2.circle(frame, px_caderas, 5, (0, 255, 255), -1)
        # Dibujo líneas pierna
        

        # Textos
        cv2.putText(frame, f"{angulo_inclinacion:.2f}", (px_caderas[0] + 10, px_caderas[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)

    output_path = os.path.join(carpeta_frame, f"Frame_{num_frame}_angulos_T_positivo.jpg")
    cv2.imwrite(output_path, frame)
    print(f"Frame guardado en: {output_path}")

    cv2.imshow("Frame con Pierna Izquierda", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    captura.release()


# Uso de la función
# analizar_rodilla_tobillo(video="ruta/a/tu/video.MOV", num_frame=165, carpeta_frame="ruta/a/carpeta/destino")


#descarga_piernas(video = "G:/Mi unidad/Videos Trabajo de Grado/Lateral Izquierda (Sofi)/Piso_LI_46.MOV",carpeta_frame = r"C:\Users\laura\OneDrive\Documents\TrabajoGrado_LauraSalamanca\Frames", num_frame = 319 )
#descarga_normal(video = r"G:/Mi unidad/Videos Trabajo de Grado/Lateral Izquierda (Sofi)/Piso_LI_17.MOV",carpeta_frame = r"C:\Users\laura\OneDrive\Documents\TrabajoGrado_LauraSalamanca\Frames", num_frame = 290 )
#descarga_normal(video = r"G:/Mi unidad/Videos Trabajo de Grado/Trasera (Andy)/Piso_T_17.MOV",carpeta_frame = r"C:\Users\laura\OneDrive\Documents\TrabajoGrado_LauraSalamanca\Frames", num_frame = 165 )
#analizar_rodilla_tobillo(video = r"G:/Mi unidad/Videos Trabajo de Grado/Lateral Derecha (Lau)/Piso_LD_17.MOV",carpeta_frame = r"C:\Users\laura\OneDrive\Documents\TrabajoGrado_LauraSalamanca\Frames", num_frame = 575)
#analizar_piernas_adelante(video = r"G:/Mi unidad/Videos Trabajo de Grado/Lateral Derecha (Lau)/Piso_LD_17.MOV",carpeta_frame = r"C:\Users\laura\OneDrive\Documents\TrabajoGrado_LauraSalamanca\Frames", num_frame = 582)

#analizar_piernas_adelante(video = r"G:/Mi unidad/Videos Trabajo de Grado/Lateral Izquierda (Sofi)/Piso_LI_17.MOV",carpeta_frame = r"C:\Users\laura\OneDrive\Documents\TrabajoGrado_LauraSalamanca\Frames", num_frame = 304)
#analizar_tobillos_z(video = r"G:/Mi unidad/Videos Trabajo de Grado/Trasera (Andy)/Piso_T_18.MOV",carpeta_frame = r"C:\Users\laura\OneDrive\Documents\TrabajoGrado_LauraSalamanca\Frames", num_frame = 104)
#analizar_angulos_derecha(video = r"G:/Mi unidad/Videos Trabajo de Grado/Lateral Derecha (Lau)/Piso_LD_17.MOV",carpeta_frame = r"C:\Users\laura\OneDrive\Documents\TrabajoGrado_LauraSalamanca\Frames", num_frame = 575)
analizar_angulos_trasera(video = r"G:/Mi unidad/Videos Trabajo de Grado/Trasera (Andy)/Piso_T_18.MOV",carpeta_frame = r"C:\Users\laura\OneDrive\Documents\TrabajoGrado_LauraSalamanca\Frames", num_frame = 50)
#dibujar_pierna_derecha(video = r"G:/Mi unidad/Videos Trabajo de Grado/Trasera (Andy)/Piso_T_12.MOV",carpeta_frame = r"C:\Users\laura\OneDrive\Documents\TrabajoGrado_LauraSalamanca\Frames", num_frame = 148)
#analizar_angulos_izquierda(video = r"G:/Mi unidad/Videos Trabajo de Grado/Lateral Izquierda (Sofi)/Piso_LI_17.MOV",carpeta_frame = r"C:\Users\laura\OneDrive\Documents\TrabajoGrado_LauraSalamanca\Frames", num_frame = 304)