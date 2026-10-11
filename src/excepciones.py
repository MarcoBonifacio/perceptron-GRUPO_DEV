"""Excepciones propias del proyecto."""


class DatosInvalidosError(Exception):
    """Señala que los datos de entrada no cumplen los requisitos."""


class ModeloNoEntrenadoError(Exception):
    """Señala que se intentó usar un modelo antes de entrenarlo."""
