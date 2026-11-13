"""Accuracy, macro-F1, matriz de confusión y tiempo de ajuste."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from src.config import ACEPTACION_CONOCIDOS, REPETICIONES_TIEMPO
from src.modelos import ResultadoClasificacion


def cronometrar(
    accion: Callable[[], Any],
    repeticiones: int = REPETICIONES_TIEMPO,
) -> tuple[Any, float]:
    """Repite ``accion`` y devuelve su último resultado con la mediana del tiempo.

    Una sola medición de milisegundos depende demasiado de lo que esté
    haciendo el equipo en ese instante.
    """
    if repeticiones < 1:
        raise ValueError("Se necesita al menos una repetición para medir.")
    tiempos = []
    for _ in range(repeticiones):
        inicio = time.perf_counter()
        resultado = accion()
        tiempos.append(time.perf_counter() - inicio)
    return resultado, float(np.median(tiempos))

def tasa_falsa_aceptacion(
    puntaje_conocidas: np.ndarray,
    puntaje_desconocidas: np.ndarray,
    aceptacion: float = ACEPTACION_CONOCIDOS,
) -> float:
    """Fracción de desconocidos aceptados con el umbral que acepta ``aceptacion`` de los conocidos.

    Un puntaje más alto significa "se parece más a alguien conocido".
    """
    umbral = np.quantile(puntaje_conocidas, 1 - aceptacion)
    return float(np.mean(puntaje_desconocidas >= umbral))




class Evaluador:
    """Ajusta el clasificador en Z_train y mide solo sobre Z_test."""

    def __init__(self, repeticiones: int = REPETICIONES_TIEMPO) -> None:
        self.repeticiones = repeticiones

    def evaluar(
        self,
        modelo: ClassifierMixin,
        Z_train: np.ndarray,
        y_train: np.ndarray,
        Z_test: np.ndarray,
        y_test: np.ndarray,
    ) -> ResultadoClasificacion:
        etiquetas = np.unique(y_train)
        # Con semilla fija, cada repetición ajusta el mismo modelo.
        predicciones, tiempo_s = cronometrar(
            lambda: np.asarray(modelo.fit(Z_train, y_train).predict(Z_test)),
            self.repeticiones,
        )
        if predicciones.shape != (y_test.shape[0],):
            raise ValueError(
                f"Se esperaban {y_test.shape[0]} predicciones y llegaron {predicciones.shape}."
            )
        matriz = confusion_matrix(y_test, predicciones, labels=etiquetas)
        return ResultadoClasificacion(
            accuracy=float(accuracy_score(y_test, predicciones)),
            macro_f1=float(f1_score(y_test, predicciones, average="macro", zero_division=0)),
            tiempo_s=tiempo_s,
            predicciones=predicciones,
            matriz=matriz,
        )
