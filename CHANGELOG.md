# Changelog

Formato: cada versión lista lo que se agregó (`Agregado`), lo que se corrigió (`Corregido`) y lo que cambió (`Cambiado`).

## [1.0.0] - 2026-10-10

### Agregado
- Métricas de evaluación: accuracy, error de clasificación y matriz de confusión.
- Decaimiento opcional de la tasa de aprendizaje del Perceptrón.
- Selección de características mediante el argumento `--features`.
- Manejo de errores para datos inválidos y validación de entradas.
- Comparación de tres modelos con respuestas analíticas en el README.

### Cambiado
- La evaluación de modelos se centralizó en `evaluar_modelo` y las épocas se configuraron con `EPOCAS`.
- La documentación describe los resultados del clasificador y el uso de sus opciones.

## [0.9.1] - 2026-10-10
### Corregido
- Evita la división entre cero al estandarizar los datos modificando `src/datos.py`.

## [0.9.0]

### Agregado
- Carga, limpieza, estandarización y división de los datos.
- Perceptrón simple con entrenamiento por épocas.
- Modelos 1 y 2 en `main.py`, evaluados con accuracy.
