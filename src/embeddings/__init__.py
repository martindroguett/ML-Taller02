"""Embeddings lineales: PCA, LDA y Fisherfaces."""

from src.embeddings.base import ExtractorEmbedding
from src.embeddings.fabrica import FabricaExtractores
from src.embeddings.fisherfaces import ExtractorFisherfaces
from src.embeddings.lda import ExtractorLDA
from src.embeddings.pca import ExtractorPCA, ExtractorPCASinPrimeras

__all__ = [
    "ExtractorEmbedding",
    "ExtractorFisherfaces",
    "ExtractorLDA",
    "ExtractorPCA",
    "ExtractorPCASinPrimeras",
    "FabricaExtractores",
]
