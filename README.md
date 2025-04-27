# Análisis biomecánico del saque de piso en arqueros utilizando estimación de pose sin marcadores

El objetivo principal es desarrollar un sistema de **análisis biomecánico** del gesto técnico del **saque de piso** en arqueros de fútbol universitario, utilizando **visión por computadora** y **estimación de pose sin marcadores** a través de **MediaPipe**.

## Descripción general

Se diseñó un flujo de procesamiento que permite:

- **Capturar videos** desde múltiples vistas (lateral derecha, lateral izquierda y trasera).
- **Detectar y extraer automáticamente los puntos clave** de las extremidades inferiores.
- **Corregir errores de estimación** de MediaPipe en frames inconsistentes.
- **Calcular ángulos biomecánicos** relevantes (rodilla, tobillo y pie) para cada fase del movimiento.
- **Comparar** los resultados con mediciones manuales obtenidas mediante **Kinovea**.
- **Validar estadísticamente** la concordancia entre métodos.
- **Visualizar resultados** a través de una **interfaz web** construida en **Django**.

