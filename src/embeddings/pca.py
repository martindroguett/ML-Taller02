"""PCA: conserva las direcciones de mayor varianza, sin usar y."""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA

from src.config import RANDOM_STATE
from src.embeddings.base import ExtractorEmbedding


class ExtractorPCA(ExtractorEmbedding):
    """Análisis de componentes principales ajustado solo con TRAIN."""

    nombre = "pca"

    def __init__(self, n_components: int, random_state: int = RANDOM_STATE) -> None:
        if n_components < 1:
            raise ValueError("PCA requiere al menos una componente.")
        self.n_components = int(n_components)
        self.random_state = random_state
        self._modelo = PCA(
            n_components=self.n_components,
            svd_solver="full",
            random_state=random_state,
        )

    @property
    def dimension(self) -> int:
        return self.n_components

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> ExtractorPCA:
        del y  # PCA no usa la identidad.
        n_max = min(X.shape[0], X.shape[1])
        if self.n_components > n_max:
            raise ValueError(
                f"PCA no puede conservar d={self.n_components} con rango máximo {n_max}."
            )
        self._modelo.fit(X)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self.comprobar(self._modelo.transform(X), X.shape[0])

    def varianza_acumulada(self) -> np.ndarray:
        """Fracción de varianza retenida por las primeras componentes."""
        return np.cumsum(self._modelo.explained_variance_ratio_)

    def componentes(self) -> np.ndarray:
        """W_PCA con una componente por fila: forma (d, p)."""
        return self._modelo.components_

    def componentes_imagen(self, alto: int, ancho: int) -> np.ndarray:
        """Reordena cada componente a H×W para verla como eigenface."""
        componentes = self._modelo.components_
        if componentes.shape[1] != alto * ancho:
            raise ValueError("La componente PCA no coincide con alto × ancho.")
        return componentes.reshape(-1, alto, ancho)

    def rostro_medio(self, alto: int, ancho: int) -> np.ndarray:
        """Media de TRAIN que sklearn resta antes de proyectar."""
        return self._modelo.mean_.reshape(alto, ancho)
