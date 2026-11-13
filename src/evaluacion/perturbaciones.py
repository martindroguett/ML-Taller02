"""Cambios sintéticos que se aplican solo a las fotos de TEST."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np


def iluminar(X: np.ndarray, alto: int, ancho: int, intensidad: float) -> np.ndarray:
    """Simula luz lateral: oscurece el lado izquierdo y aclara el derecho.

    Cada píxel se multiplica por 1 + intensidad·g, con g de −1 a 1 de
    izquierda a derecha. Con intensidad 0 la foto no cambia.
    """
    if intensidad < 0:
        raise ValueError("La intensidad de la luz no puede ser negativa.")
    gradiente = np.tile(np.linspace(-1.0, 1.0, ancho), alto)
    return np.clip(X * (1 + intensidad * gradiente), 0.0, 1.0)


def ocluir(
    X: np.ndarray,
    alto: int,
    ancho: int,
    fila: int,
    columna: int,
    tamano: int,
) -> np.ndarray:
    """Pone en negro un cuadrado de ``tamano`` píxeles desde (fila, columna)."""
    mascara = np.ones((alto, ancho))
    mascara[fila : fila + tamano, columna : columna + tamano] = 0
    return X * mascara.ravel()


def ventanas(alto: int, ancho: int, tamano: int) -> Iterator[tuple[int, int]]:
    """Esquina superior izquierda de cada cuadrado de una grilla que cubre la imagen."""
    for fila in range(0, alto, tamano):
        for columna in range(0, ancho, tamano):
            yield fila, columna
