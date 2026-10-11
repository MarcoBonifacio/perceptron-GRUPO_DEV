"""Métricas para evaluar un clasificador binario."""
import numpy as np


def accuracy(y, y_pred):
    """Proporción de predicciones correctas (entre 0 y 1)."""
    return float(np.mean(np.asarray(y) == np.asarray(y_pred)))


def error_clasificacion(y, y_pred):
    """Proporción de predicciones incorrectas (entre 0 y 1)."""
    return 1 - np.mean(np.asarray(y) == np.asarray(y_pred))

def matriz_confusion(y, y_pred):
    """Devuelve la matriz 2x2 [[TN, FP], [FN, TP]] como array de NumPy."""
    tn = int(np.sum((y == 0) & (y_pred == 0)))
    fp = int(np.sum((y == 0) & (y_pred == 1)))
    fn = int(np.sum((y == 1) & (y_pred == 0)))
    tp = int(np.sum((y == 1) & (y_pred == 1)))
    return [[tn, fp], [fn, tp]]