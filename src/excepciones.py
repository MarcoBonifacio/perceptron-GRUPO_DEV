"Excepciones del proyecto de perceptrón."


class DatosInvalidosError(ValueError):
    "Se lanza cuando los datos de entrada no cumplen el formato esperado."

    pass


class ModeloNoEntrenadoError(RuntimeError):
    "Se lanza cuando se intenta usar un modelo que aún no fue entrenado."

    pass
