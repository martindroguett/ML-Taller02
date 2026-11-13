"""Construcción de PCA, LDA y Fisherfaces para el mismo protocolo."""

from __future__ import annotations

from src.config import D_EXTRAS, D_PCA_VARIANTE, DIMENSIONES
from src.embeddings.base import ExtractorEmbedding
from src.embeddings.fisherfaces import ExtractorFisherfaces
from src.embeddings.lda import ExtractorLDA
from src.embeddings.pca import ExtractorPCA, ExtractorPCASinPrimeras


class FabricaExtractores:
    """Crea un extractor por nombre o la lista completa del experimento."""

    def crear(self, nombre: str, d: int, n_clases: int) -> ExtractorEmbedding:
        if nombre == "pca":
            return ExtractorPCA(n_components=d)
        if nombre == "lda":
            return ExtractorLDA(n_components=self.dimension_lda(d, n_clases))
        if nombre == "fisherfaces":
            return ExtractorFisherfaces(
                d_pca=d,
                d_lda=self.dimension_lda(d, n_clases),
            )
        raise ValueError(f"Embedding desconocido: {nombre}")

    def para_experimento(self, n_clases: int) -> list[ExtractorEmbedding]:
        """PCA, LDA y Fisherfaces en d = 10, 20, 30."""
        extractores: list[ExtractorEmbedding] = []
        lda_vistos: set[int] = set()
        for d in DIMENSIONES:
            extractores.append(ExtractorPCA(n_components=d))
            d_lda = self.dimension_lda(d, n_clases)
            if d_lda not in lda_vistos:
                lda_vistos.add(d_lda)
                extractores.append(ExtractorLDA(n_components=d_lda))
            extractores.append(ExtractorFisherfaces(d_pca=d, d_lda=d_lda))
        return extractores

    def variantes_fisherfaces(self, n_clases: int, n_train: int) -> list[ExtractorEmbedding]:
        """Fisherfaces con más componentes PCA que LDA; no son parte del protocolo.

        Una variante fija d_pca = 100 y la otra usa d_pca = N − C, el valor
        del método original de Belhumeur et al. (1997).
        """
        opciones = {
            f"fisherfaces_pca{D_PCA_VARIANTE}": D_PCA_VARIANTE,
            "fisherfaces_nc": n_train - n_clases,
        }
        extractores: list[ExtractorEmbedding] = []
        for nombre, d_pca in opciones.items():
            for d in DIMENSIONES:
                d_lda = self.dimension_lda(d, n_clases)
                extractores.append(ExtractorFisherfaces(d_pca=d_pca, d_lda=d_lda, nombre=nombre))
        return extractores

    def para_extras(self, n_clases: int, d: int = D_EXTRAS) -> list[ExtractorEmbedding]:
        """Embeddings que compara el análisis extendido, todos con d coordenadas."""
        d_lda = self.dimension_lda(d, n_clases)
        return [
            ExtractorPCA(n_components=d),
            ExtractorPCASinPrimeras(n_components=d),
            ExtractorLDA(n_components=d_lda),
            ExtractorFisherfaces(d_pca=d, d_lda=d_lda),
            ExtractorFisherfaces(
                d_pca=D_PCA_VARIANTE,
                d_lda=d_lda,
                nombre=f"fisherfaces_pca{D_PCA_VARIANTE}",
            ),
        ]



    @staticmethod
    def dimension_lda(d: int, n_clases: int) -> int:
        """LDA no puede superar C − 1 ni pedir más coordenadas que las de entrada."""
        return max(1, min(d, n_clases - 1))
