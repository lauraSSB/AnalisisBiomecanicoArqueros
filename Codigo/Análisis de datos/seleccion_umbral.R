library(ggplot2)
library(readr)
library(dplyr)

df <- read_csv("G:/Mi unidad/Videos Trabajo de Grado/data_frame_distancias_corregido_T.csv")

df_filtrado <- df %>% filter(marcador == "rodilla_izquierda")

#cadera derecha "#66C2A5" 22
#hombro derecho FC8D62 18
#rodilla derecha 8DA0CB 60
#punta_derecha E78AC3 150
#tobillo_derecho A6D854 130
#talon derecho FFD92F 150
#hombro izqu, cadera_izquierda 30
#rodilla 25

x_umbral <- 18
color_relleno <- "#FFF8DC"


ggplot(df_filtrado, aes(x = distancia)) +
  geom_histogram(aes(y = ..density..), bins = 30, fill = color_relleno, color = "black", alpha = 0.6) +
  geom_density(color = "red", size = 1.2) +
  geom_vline(xintercept = x_umbral, color = "blue", linetype = "dashed", size = 1) +
  labs(
    x = "Distancia entre frames (px)",
    y = "Densidad"
  ) +
  theme_minimal()


# Boxplot para análisis de atípicos
ggplot(df_filtrado, aes(x = distancia)) +
  geom_boxplot(fill = color_relleno, outlier.color = "red", outlier.size = 2) +
  geom_vline(xintercept = x_umbral, color = "blue", linetype = "dashed", size = 1) +
  labs(
    x = "Distancia entre frames (px)"
  ) +
  theme_minimal() 
  

