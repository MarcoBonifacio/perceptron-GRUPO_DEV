"""Carga y preparación de los datos para el Perceptrón."""
import numpy as np
import pandas as pd


def cargar_datos(ruta, objetivo):
    """Lee el CSV y devuelve el DataFrame.

    `objetivo` es el nombre de la columna con la clase (0 o 1).
    """
    df = pd.read_csv(ruta)
    return df


def limpiar(df, features):
    """Devuelve una copia con los nulos de cada feature rellenados con su mediana."""
    datos = df.copy()
    for col in features:
        datos[col] = datos[col].fillna(datos[col].median())
    return datos


def estandarizar(X):
    """Estandariza los datos restando la media y dividiendo por la desviación estándar."""
    media = np.mean(X, axis=0)
    desv = np.std(X, axis=0)

    # Evitar división entre cero
    desv[desv == 0] = 1

    return (X - media) / desv


def dividir(X, y, prueba=0.2, semilla=42):
    """Mezcla las filas y separa entrenamiento y prueba."""
    idx = np.random.default_rng(semilla).permutation(len(X))
    n_prueba = int(len(X) * prueba)
    te, tr = idx[:n_prueba], idx[n_prueba:]
    return X[tr], X[te], y[tr], y[te]
