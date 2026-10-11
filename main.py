"""Entrena y evalúa los modelos de Perceptrón.

Uso:
    python main.py --datos datos/pacientes.csv --objetivo diagnostico
"""
import argparse
import sys

from src.datos import cargar_datos, limpiar, estandarizar, dividir
from src.excepciones import DatosInvalidosError
from src.perceptron import Perceptron
from src.metricas import accuracy, error_clasificacion, matriz_confusion

FEATURES_BASE = ["radio", "textura", "perimetro", "area"]
FEATURES_MODELO3 = ["concavidad", "puntos_concavos", "area", "textura"]
EPOCAS = 30


def evaluar_modelo(nombre, df, features, objetivo, modelo):
    """Entrena el modelo y muestra sus métricas de evaluación."""
    columnas_inexistentes = [feature for feature in features if feature not in df.columns]
    if columnas_inexistentes:
        raise DatosInvalidosError(
            f"Columnas inexistentes: {', '.join(columnas_inexistentes)}"
        )

    datos = limpiar(df, features)
    X = estandarizar(datos[features].to_numpy(dtype=float))
    y = datos[objetivo].to_numpy()
    X_tr, X_te, y_tr, y_te = dividir(X, y)
    modelo.entrenar(X_tr, y_tr)
    y_pred = modelo.predecir(X_te)

    print(nombre)
    print("  accuracy:", round(accuracy(y_te, y_pred), 3))
    print("  errores por época:", modelo.errores_por_epoca[:10], "...")
    print(f"Error de clasificación: {error_clasificacion(y_te, y_pred):.3f}")
    print(f"Matriz de confusión:\n{matriz_confusion(y_te, y_pred)}")


def main():
    parser = argparse.ArgumentParser(description="Clasificador con Perceptrón")
    parser.add_argument("--datos", default="datos/pacientes.csv")
    parser.add_argument("--objetivo", default="diagnostico")
    parser.add_argument(
        "--features",
        nargs="+",
        default=FEATURES_MODELO3,
        help="Lista de características a utilizar en el Modelo 3",
    )
    args = parser.parse_args()

    df = cargar_datos(args.datos, args.objetivo)
    # ---------------- Modelo 1: básico ----------------
    modelo1 = Perceptron(tasa_aprendizaje=0.01, epocas=EPOCAS)
    evaluar_modelo("Modelo 1 (tasa 0.01)", df, FEATURES_BASE, args.objetivo, modelo1)

    # ---------------- Modelo 2: tasa grande ----------------
    modelo2 = Perceptron(tasa_aprendizaje=0.5, epocas=EPOCAS, decaimiento=0.1)
    evaluar_modelo(
        "\nModelo 2 (tasa 0.5, decaimiento 0.1)",
        df,
        FEATURES_BASE,
        args.objetivo,
        modelo2,
    )

    # ---------------- Modelo 3: features seleccionadas ----------------
    modelo3 = Perceptron(tasa_aprendizaje=0.01, epocas=EPOCAS)
    evaluar_modelo(
        "\nModelo 3 (features seleccionadas)",
        df,
        args.features,
        args.objetivo,
        modelo3,
    )


if __name__ == "__main__":
    try:
        main()
    except (DatosInvalidosError, ValueError) as error:
        print(f"Error: {error}")
        sys.exit(1)
