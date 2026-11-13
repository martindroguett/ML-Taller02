"""LDA: usa las etiquetas para separar identidades."""

from __future__ import annotations

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from src.embeddings.base import ExtractorEmbedding


class ExtractorLDA(ExtractorEmbedding):
    """Análisis discriminante lineal. d no puede superar C − 1."""

    nombre = "lda"

    def __init__(self, n_components: int) -> None:
        if n_components < 1:
            raise ValueError("LDA requiere al menos una componente.")
        self.n_components = int(n_components)
        self._modelo = LinearDiscriminantAnalysis(
            n_components=self.n_components,
            solver="svd",
        )

    @property
    def dimension(self) -> int:
        return self.n_components

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> ExtractorLDA:
        if y is None:
            raise ValueError("LDA necesita y_train: aprende con las identidades.")
        n_clases = int(np.unique(y).size)
        maximo = n_clases - 1
        if self.n_components > maximo:
            raise ValueError(
                f"LDA con d={self.n_components} supera C-1={maximo}. Reduzca d."
            )
        self._modelo.fit(X, y)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self.comprobar(self._modelo.transform(X), X.shape[0])

    def direcciones(self) -> np.ndarray:
        """W_LDA con una dirección discriminante por columna: forma (p, d)."""
        return self._modelo.scalings_[:, : self.n_components]
