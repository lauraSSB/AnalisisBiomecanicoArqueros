library(readr)
library(ggplot2)
library(nortest)
library(stats)

datos <- read_csv("C:/Users/laura/OneDrive/Documents/TrabajoGrado_LauraSalamanca/comparacion_T_final.csv")

summary(datos$diferencia_rodilla_izquierda)

ggplot(datos, aes(x = diferencia_rodilla_izquierda)) +
  geom_histogram(binwidth = 1, fill = "#FFB5C5", color = "black") +
  geom_vline(xintercept = 0, color = "red", linetype = "dashed", linewidth = 1) +
  xlim(-50, 150) +
  labs(x = "Diferencia rodilla",
       y = "Frecuencia") +
  theme_minimal()

ggplot(datos, aes(x = diferencia_rodilla_izquierda)) +
  geom_boxplot(fill = "#FFB5C5", outlier.color = "red", outlier.shape = 18, outlier.size = 3) +
  geom_vline(xintercept = 0, color = "blue", linetype = "dashed", linewidth = 1) +
  labs(x = "Diferencia rodilla") +
  theme_minimal() +
  theme(
    axis.text.x = element_text(size = 14)
  )

q1 <- quantile(datos$diferencia_rodilla_izquierda, 0.25, na.rm = TRUE)
q3 <- quantile(datos$diferencia_rodilla_izquierda, 0.75, na.rm = TRUE)
iqr <- q3 - q1

lim_inf <- q1 - 1.5 * iqr
lim_sup <- q3 + 1.5 * iqr

outliers <- datos[datos$diferencia_rodilla_izquierda < lim_inf | datos$diferencia_rodilla_izquierda > lim_sup, ]

datos_limpios <- datos[datos$diferencia_rodilla_izquierda >= lim_inf & datos$diferencia_rodilla_izquierda <= lim_sup, ]

nrow(datos) - nrow(datos_limpios)

summary(datos_limpios$diferencia_rodilla_izquierda)

ggplot(datos_limpios, aes(x = diferencia_rodilla_izquierda)) +
  geom_histogram(binwidth = 1, fill = "#FFB5C5", color = "black") +
  geom_vline(xintercept = 0, color = "red", linetype = "dashed", linewidth = 2) +
  geom_vline(xintercept = mean(datos_limpios$diferencia_rodilla_izquierda), color = "#fb03d5", linetype = "dashed", linewidth = 2) +
  xlim(-20, 20) +
  labs(x = "Diferencia rodilla",
       y = "Frecuencia") +
  theme_minimal() +
  theme(
    axis.text.x = element_text(size = 14)
  )

ggplot(datos_limpios, aes(x = diferencia_rodilla_izquierda)) +
  geom_boxplot(fill = "#FFB5C5", outlier.color = "red", outlier.shape = 18, outlier.size = 3) +
  geom_vline(xintercept = 0, color = "red", linetype = "dashed", linewidth = 2) +
  geom_vline(xintercept = mean(datos_limpios$diferencia_rodilla_izquierda), color = "#fb03d5", linetype = "dashed", linewidth = 2) +
  labs(y = "Diferencia rodilla") +
  theme_minimal()

shapiro.test(datos_limpios$diferencia_rodilla_izquierda)
lillie.test(datos_limpios$diferencia_rodilla_izquierda)

boxplot(datos_limpios$angulo_rodilla_kinovea, datos_limpios$angulo_rodilla_mediapipe,
        names = c("Ángulo rodilla Kinovea", "Ángulo rodilla MediaPipe"),
        main = "Boxplots de las dos variables por separado",
        ylab = "Valores",
        col = c("skyblue", "lightgreen"))

t.test(datos_limpios$diferencia_rodilla_izquierda)
library(BSDA)

SIGN.test(datos_limpios$angulo_rodilla_kinovea, datos_limpios$angulo_rodilla_mediapipe, alternative = "two.sided",paired = TRUE)
correlacion <- cor(datos_limpios$angulo_rodilla_izquierda_kinovea, datos_limpios$angulo_rodilla_izquierda_mediapipe)
ggplot(datos_limpios, aes(x = angulo_rodilla_izquierda_kinovea, y = angulo_rodilla_izquierda_mediapipe)) +
  geom_point(alpha = 0.7, color = "steelblue") +
  geom_abline(intercept = 0, slope = 1, color = "red", linewidth = 1.5) +
  labs(
    x = "Ángulo rodilla Kinovea (°)",
    y = "Ángulo rodilla MediaPipe (°)"
  ) +
  theme_minimal()

)#---------------------------------------------------------------------------------------------------------------------------

summary(datos$diferencia_pie)

ggplot(datos, aes(x = diferencia_pie)) +
  geom_histogram(binwidth = 1, fill = "#C1FFC1", color = "black") +
  geom_vline(xintercept = 0, color = "red", linetype = "dashed", linewidth = 1) +
  xlim(-150, 150) +
  labs(x = "Diferencia pie",
       y = "Frecuencia") +
  theme_minimal()

ggplot(datos, aes(x = diferencia_pie)) +
  geom_boxplot(fill = "#C1FFC1", outlier.color = "red", outlier.shape = 18, outlier.size = 3) +
  geom_vline(xintercept = 0, color = "blue", linetype = "dashed", linewidth = 1) +
  labs(x = "Diferencia pie") +
  theme_minimal() +
  theme(
    axis.text.x = element_text(size = 14)
  )

q1_pie <- quantile(datos$diferencia_pie, 0.25, na.rm = TRUE)
q3_pie <- quantile(datos$diferencia_pie, 0.75, na.rm = TRUE)
iqr_pie <- q3_pie - q1_pie

lim_inf_pie <- q1_pie - 1.5 * iqr_pie
lim_sup_pie <- q3_pie + 1.5 * iqr_pie

outliers_pie <- datos[datos$diferencia_pie < lim_inf_pie | datos$diferencia_pie > lim_sup_pie, ]

datos_limpios_pie <- datos[datos$diferencia_pie >= lim_inf_pie & datos$diferencia_pie <= lim_sup_pie, ]

nrow(datos) - nrow(datos_limpios_pie)

summary(datos_limpios_pie$diferencia_pie)

ggplot(datos_limpios_pie, aes(x = diferencia_pie)) +
  geom_histogram(binwidth = 1, fill = "#C1FFC1", color = "black") +
  geom_vline(xintercept = 0, color = "red", linetype = "dashed", linewidth = 1) + 
  geom_vline(xintercept = mean(datos_limpios_pie$diferencia_pie), color = "#fb03d5", linetype = "dashed", linewidth = 2) +
  xlim(-45, 45) +
  labs(x = "Diferencia pie",
       y = "Frecuencia") +
  theme_minimal()


ggplot(datos_limpios_pie, aes(x = diferencia_pie)) +
  geom_boxplot(fill = "#C1FFC1", outlier.color = "red", outlier.shape = 18, outlier.size = 3) +
  geom_vline(xintercept = 0, color = "blue", linetype = "dashed", linewidth = 1) +
  geom_vline(xintercept = mean(datos_limpios_pie$diferencia_pie), color = "#fb03d5", linetype = "dashed", linewidth = 2) +
  labs(x = "Diferencia pie") +
  theme_minimal() +
  theme(
    axis.text.x = element_text(size = 14)
  )

shapiro.test(datos_limpios_pie$diferencia_pie)
lillie.test(datos_limpios_pie$diferencia_pie)

t.test(datos_limpios_pie$diferencia_pie)
SIGN.test(datos_limpios_pie$angulo_pie_kinovea, datos_limpios_pie$angulo_pie_mediapipe, alternative = "two.sided",paired = TRUE)
correlacion <- cor(datos_limpios_pie$angulo_pie_kinovea, datos_limpios_pie$angulo_pie_mediapipe)
ggplot(datos_limpios_pie, aes(x = angulo_pie_kinovea, y = angulo_pie_mediapipe)) +
  geom_point(alpha = 0.7, color = "steelblue") +
  geom_abline(intercept = 0, slope = 1, color = "red", linewidth = 1.5) +
  labs(
    x = "Ángulo pie Kinovea (°)",
    y = "Ángulo pie MediaPipe (°)"
  ) +
  theme_minimal()
