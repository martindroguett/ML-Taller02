"""Carga de LFW con las identidades más fotografiadas."""

from __future__ import annotations

import numpy as np
from sklearn.datasets import fetch_lfw_people

from src.config import DIR_CACHE, LFW_MAX_IDENTIDADES, LFW_MIN_FOTOS, LFW_RESIZE
from src.datos.base import CargadorRostros, validar_conjunto
from src.modelos import ConjuntoRostros


class CargadorLFW(CargadorRostros):
    """Fotografías de LFW con mayor diversidad de pose, luz y fondo.

    Se piden las identidades con al menos 20 fotos y se conservan, como
    máximo, las 50 con más imágenes. Las etiquetas quedan en 0..C−1 y los
    píxeles vienen en [0, 1], igual que en Olivetti.
    """

    nombre = "lfw"

    def __init__(
        self,
        min_fotos: int = LFW_MIN_FOTOS,
        max_identidades: int = LFW_MAX_IDENTIDADES,
    ) -> None:
        self.min_fotos = min_fotos
        self.max_identidades = max_identidades

    def cargar(self) -> ConjuntoRostros:
        rostros = fetch_lfw_people(
            min_faces_per_person=self.min_fotos,
            resize=LFW_RESIZE,
            data_home=str(DIR_CACHE),
            download_if_missing=True,
        )
        mascara, y, nombres = self.seleccionar_identidades(
            np.asarray(rostros.target, dtype=int),
            [str(nombre) for nombre in rostros.target_names],
            self.max_identidades,
        )
        images = np.asarray(rostros.images)[mascara]
        X = np.asarray(rostros.data, dtype=float)[mascara]
        alto, ancho = int(images.shape[1]), int(images.shape[2])
        conjunto = ConjuntoRostros(
            nombre=self.nombre,
            images=images,
            X=X,
            y=y,
            nombres_identidades=nombres,
            alto=alto,
            ancho=ancho,
        )
        validar_conjunto(conjunto)
        return conjunto

    @staticmethod
    def seleccionar_identidades(
        y: np.ndarray,
        nombres: list[str],
        maximo: int,
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Elige las identidades con más fotos y reindexa las etiquetas.

        Devuelve la máscara de fotos conservadas, las etiquetas nuevas en
        0..C−1 y los nombres en ese mismo orden. Si hay empate en el corte
        entra el índice original más bajo, que en LFW es el orden alfabético.
        """
        conteos = np.bincount(y, minlength=len(nombres))
        elegidas = np.argsort(-conteos, kind="stable")[:maximo]
        mascara = np.isin(y, elegidas)
        originales, y_nuevo = np.unique(y[mascara], return_inverse=True)
        return mascara, y_nuevo, [nombres[i] for i in originales]
