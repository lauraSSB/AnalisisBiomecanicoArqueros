import cv2
import mediapipe as mp

# Inicializar Mediapipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils  # Herramienta para dibujar
pose = mp_pose.Pose(min_detection_confidence=0.90, min_tracking_confidence=0.95, model_complexity=2)

# Capturar video desde el archivo
cap = cv2.VideoCapture("G:/Mi unidad/Videos Trabajo de Grado/Trasera (Andy)/Piso_T_11.MOV")  

# Variable de pausa
pausa = False  

while cap.isOpened():
    if not pausa:  # Solo leer un nuevo frame si no está en pausa
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.rotate(frame, cv2.ROTATE_180)
        frame = cv2.resize(frame, (int(frame.shape[1] * 0.6), int(frame.shape[0] * 0.6)))

        # Convertir a RGB para Mediapipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Procesar el frame con Mediapipe Pose
        results = pose.process(frame_rgb)   

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            h, w, _ = frame.shape  # Obtener dimensiones de la imagen

            # Obtener coordenadas de las caderas
            right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
            left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]

            # Calcular punto medio de las caderas
            hip_center_x = int((right_hip.x + left_hip.x) / 2 * w)
            hip_center_y = int((right_hip.y + left_hip.y) / 2 * h)

            # Calcular coordenada Z de la mitad de la cadera
            hip_center_z = (right_hip.z + left_hip.z) / 2

            # Obtener coordenadas de las puntas de los pies
            right_foot = landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value]
            left_foot = landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value]

            # Convertir coordenadas normalizadas a píxeles
            right_foot_x, right_foot_y = int(right_foot.x * w), int(right_foot.y * h)
            left_foot_x, left_foot_y = int(left_foot.x * w), int(left_foot.y * h)

            # Dibujar los puntos en la imagen
            cv2.circle(frame, (hip_center_x, hip_center_y), 3, (0, 255, 0), -1)  # Punto medio de caderas (verde)
            cv2.circle(frame, (right_foot_x, right_foot_y), 3, (255, 0, 0), -1)  # Pie derecho (azul)
            cv2.circle(frame, (left_foot_x, left_foot_y), 3, (255, 0, 0), -1)  # Pie izquierdo (azul)

            # Dibujar etiquetas
            cv2.putText(frame, "Hip Center", (hip_center_x, hip_center_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(frame, "Right Foot", (right_foot_x, right_foot_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            cv2.putText(frame, "Left Foot", (left_foot_x, left_foot_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

            # Imprimir coordenadas Z en la consola
            print(f"Z Mitad Cadera: {hip_center_z:.4f}")
            print(f"Z Pie Derecho : {right_foot.z:.4f}")
            print(f"Z Pie Izquierdo: {left_foot.z:.4f}")
            print("-" * 50)

    # Mostrar el frame con los puntos dibujados
    cv2.imshow("Pose Detection", frame)

    # Capturar la tecla presionada
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):  # Salir con "q"
        break
    elif key == ord('p'):  # Pausar con "p"
        pausa = not pausa  # Cambiar el estado de pausa

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
