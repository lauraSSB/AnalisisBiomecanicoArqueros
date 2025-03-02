import cv2
import os

imagen = cv2.imread("./Frames/frame3.jpg") 
carpeta_frame = r"C:\Users\laura\OneDrive\Documents\TrabajoGrado_LauraSalamanca\Frames"

os.makedirs(carpeta_frame, exist_ok=True)


if imagen is None:
    print("Error al cargar la imagen")
    exit()

imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)  
output_path = os.path.join(carpeta_frame, "frame3_RGB.jpg")
cv2.imwrite(output_path, imagen_rgb)
imagen_hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)  
output_path = os.path.join(carpeta_frame, "frame3_HSV.jpg")
cv2.imwrite(output_path, imagen_hsv)
imagen_ycrcb = cv2.cvtColor(imagen, cv2.COLOR_BGR2YCrCb)
output_path = os.path.join(carpeta_frame, "frame3_YCrCb.jpg")
cv2.imwrite(output_path, imagen_ycrcb)
imagen_lab = cv2.cvtColor(imagen, cv2.COLOR_BGR2LAB)  
output_path = os.path.join(carpeta_frame, "frame3_LAB.jpg")
cv2.imwrite(output_path, imagen_lab)

cv2.imshow("Imagen en BGR (Original)", imagen)
cv2.imshow("Imagen en RGB", imagen_rgb)
cv2.imshow("Imagen en HSV", imagen_hsv)
cv2.imshow("Imagen en YCrCb", imagen_ycrcb)
cv2.imshow("Imagen en LAB", imagen_lab)

cv2.waitKey(0)
cv2.destroyAllWindows()
