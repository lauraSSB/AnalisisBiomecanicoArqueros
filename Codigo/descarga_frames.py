import cv2
import os
import mediapipe as mp

#Función que descarga un 5 frames siguientes a un determinado frame de un video. Se puede especificar la rita de descarga
def descarga_normal(video, carpeta_frame, num_frame):
    os.makedirs(carpeta_frame, exist_ok=True)

    output_path = os.path.join(carpeta_frame, "frame2.jpg")

    captura = cv2.VideoCapture(video)

    if not captura.isOpened():
        print("Error al abrir el video.")
        exit()

    for i in range(0,5):
        print(num_frame+i)
        captura.set(cv2.CAP_PROP_POS_FRAMES, num_frame+i)

        ret, frame = captura.read()
        if frame[0, 0][0] < frame[-1, -1][0]:  # Comparación de color en esquinas
            print("🔄 Video detectado con rotación 180°, corrigiendo...")
            frame = cv2.rotate(frame, cv2.ROTATE_180)  # Rota la imagen 180°

        if ret:
            cv2.imshow("Frame Extraído", frame)

            #cv2.imwrite(output_path, frame)
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
    pose = mp_pose.Pose(min_detection_confidence=0.5, model_complexity=2)
    
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
        #frame = cv2.rotate(frame, cv2.ROTATE_180)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resultado = pose.process(rgb_frame)
        
        if resultado.pose_landmarks:
            dibujar_piernas(frame, resultado.pose_landmarks.landmark, mp_pose)
        
        output_path = os.path.join(carpeta_frame, f"T_inicio{i}.jpg")
        cv2.imwrite(output_path, frame)
        #cv2.imshow("Frame",frame)
        print(f"Frame guardado en: {output_path}")
    
    captura.release()

descarga_piernas(video = "G:/Mi unidad/Videos Trabajo de Grado/Lateral Izquierda (Sofi)/Piso_LI_11.MOV",carpeta_frame = r"C:\Users\laura\OneDrive\Documents\TrabajoGrado_LauraSalamanca\Frames", num_frame = 358 )
#descarga_normal(video = r"C:\Users\laura\OneDrive - Pontificia Universidad Javeriana\Videos Tesis\Saques de Piso\Trasera(Andy)\Piso_T_5.MOV",carpeta_frame = r"C:\Users\laura\OneDrive\Documents\TrabajoGrado_LauraSalamanca\Frames", num_frame = 120 )
