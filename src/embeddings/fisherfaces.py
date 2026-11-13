"""Fisherfaces: PCA comprime y LDA discrimina sobre ese espacio."""

from __future__ import annotations

import numpy as np

from src.embeddings.base import ExtractorEmbedding
from src.embeddings.lda import ExtractorLDA
from src.embeddings.pca import ExtractorPCA


class ExtractorFisherfaces(ExtractorEmbedding):
    """Proyección z = W_LDAᵀ W_PCAᵀ (x − μ), ajustada solo con TRAIN.

    1. PCA con ``d_pca`` se ajusta solo sobre ``X_train`` (ahí se estima μ).
    2. LDA con ``d_lda`` se ajusta sobre ese PCA de TRAIN, usando ``y_train``.
    3. TEST pasa por las dos proyecciones, sin volver a ajustar.

    Contrato: ``d_lda <= min(C - 1, d_pca)`` y ``d_pca <= N - C``. Con más
    componentes PCA que N − C, la dispersión dentro de las clases queda
    singular y LDA pierde sentido.
    """

    nombre = "fisherfaces"

    def __init__(self, d_pca: int, d_lda: int, nombre: str | None = None) -> None:
        if d_pca < 1 or d_lda < 1:
            raise ValueError("Fisherfaces requiere d_pca y d_lda positivos.")
        if d_lda > d_pca:
            raise ValueError(
                f"d_lda={d_lda} supera d_pca={d_pca}. "
                "Debe cumplirse d_lda <= min(C-1, d_pca)."
            )
        self.d_pca = int(d_pca)
        self.d_lda = int(d_lda)
        if nombre:
            self.nombre = nombre
        self._pca = ExtractorPCA(n_components=self.d_pca)
        self._lda = ExtractorLDA(n_components=self.d_lda)

    @property
    def dimension(self) -> int:
        return self.d_lda

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> ExtractorFisherfaces:
        if y is None:
            raise ValueError("Fisherfaces necesita y_train: LDA aprende con las identidades.")
        maximo = X.shape[0] - int(np.unique(y).size)
        if self.d_pca > maximo:
            raise ValueError(
                f"Fisherfaces con d_pca={self.d_pca} supera N-C={maximo}. Reduzca d_pca."
            )
        self._pca.fit(X)
        self._lda.fit(self._pca.transform(X), y)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        Z_pca = self._pca.transform(X)
        return self.comprobar(self._lda.transform(Z_pca), X.shape[0])

    def componentes_imagen(self, alto: int, ancho: int) -> np.ndarray:
        """Lleva cada dirección discriminante a píxeles para verla como fisherface."""
        fisherfaces = self._lda.direcciones().T @ self._pca.componentes()
        if fisherfaces.shape[1] != alto * ancho:
            raise ValueError("La fisherface no coincide con alto × ancho.")
        return fisherfaces.reshape(-1, alto, ancho)
