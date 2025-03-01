import cv2

# Cargar la imagen
imagen = cv2.imread("imagenpruebapatada.png") 


if imagen is None:
    print("Error al cargar la imagen")
    exit()

imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)  
imagen_hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)  
imagen_ycrcb = cv2.cvtColor(imagen, cv2.COLOR_BGR2YCrCb) 
imagen_lab = cv2.cvtColor(imagen, cv2.COLOR_BGR2LAB)  

cv2.imshow("Imagen en BGR (Original)", imagen)
cv2.imshow("Imagen en RGB", imagen_rgb)
cv2.imshow("Imagen en HSV", imagen_hsv)
cv2.imshow("Imagen en YCrCb", imagen_ycrcb)
cv2.imshow("Imagen en LAB", imagen_lab)

cv2.waitKey(0)
cv2.destroyAllWindows()
