import cv2
import mediapipe as mp
import numpy as np

print("Hola")
captura = cv2.VideoCapture("C:/Users/laura/OneDrive - Pontificia Universidad Javeriana/Videos Tesis/Saques de Piso/Trasera(Andy)/Piso_T_5.MOV",cv2.CAP_FFMPEG)
prevCircle = None

dist = lambda x1,y1,x2,y2: (x1-x2)**2 + (y1-y2)**2

while True: 
    ret, frame = captura.read()
    if not ret:
        break

    grayFrame = cv.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    blurFrame = cv.GaussianBlur(grayFrame,(17,17),0)

    circles = cv.HoughCircles(blurFrame,cv.HOUGH_GRADIENT,1.2,100,
        param1=100, param3=30, minRadius=75,maxRadius=400)

    if circles is not None:
        circles = numpy.uint16(numpy.around(circles))
        chosen = None
        for i in circles[0,:]:
            if chosen is None: chosen = 1
            if prevCircle is not None:
                if dist(chosen[0],chosen[1],prevCircle[0],prevCircle[1]) <= dist(i[0],i[1],prevCircle[0],prevCircle[1]):
                    chosen = i

        cv2.circle(frame, (chosen[0],chosen[1]), 1, (0,100,100),3)
        cv2.circle(frame, (chosen[0],chosen[1]), chosen[2], (255,0,255),3)
        prevCircle = chosen

    cv2.imshow("circles",frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

captura.release()
cv2.destroyAllWindows()