"""Contratos de datos del laboratorio."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ConjuntoRostros:
    """Fotografías, matriz de píxeles y etiquetas de identidad, alineadas."""

    nombre: str
    images: np.ndarray
    X: np.ndarray
    y: np.ndarray
    nombres_identidades: list[str]
    alto: int
    ancho: int

    @property
    def n_clases(self) -> int:
        return int(np.unique(self.y).size)

    @property
    def desbalanceado(self) -> bool:
        """La identidad con más fotos tiene al menos el doble que la de menos."""
        _, conteos = np.unique(self.y, return_counts=True)
        return bool(conteos.max() >= 2 * conteos.min())


@dataclass
class Particion:
    """TRAIN y TEST disjuntos. Las imágenes siguen el mismo orden que X e y."""

    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    images_train: np.ndarray
    images_test: np.ndarray
    indices_train: np.ndarray
    indices_test: np.ndarray


@dataclass
class ResultadoExperimento:
    """Una fila de la tabla de combinaciones."""

    dataset: str
    embedding: str
    d: int
    modelo: str
    parametros: str
    accuracy: float
    macro_f1: float
    tiempo_s: float
    tiempo_embedding_s: float


@dataclass
class ResultadoClasificacion:
    """Métricas de un clasificador ya ajustado sobre el embedding de TRAIN."""

    accuracy: float
    macro_f1: float
    tiempo_s: float
    predicciones: np.ndarray
    matriz: np.ndarray


@dataclass
class Prediccion:
    """Identidad asignada a una fotografía que no entró al entrenamiento."""

    indice: int
    identidad_real: int
    identidad_predicha: int
    nombre_real: str
    nombre_predicho: str
    probabilidad: float
    acierto: bool
    top: list[tuple[str, float]] = field(default_factory=list)
    nota_probabilidad: str = ""
