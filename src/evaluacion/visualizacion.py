"""Figuras del experimento: curva PCA, eigenfaces, fisherfaces, confusión y mapas 2D."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from src.config import (
    DIMENSIONES,
    DIR_FIGURAS,
    RANDOM_STATE,
    TSNE_PERPLEXITY,
    UMAP_MIN_DIST,
    UMAP_N_NEIGHBORS,
)


class Visualizador:
    """Guarda la evidencia gráfica de TRAIN y de la evaluación en TEST."""

    def __init__(self, directorio: Path = DIR_FIGURAS) -> None:
        self.directorio = directorio
        self.directorio.mkdir(parents=True, exist_ok=True)

    def curva_pca(self, X_train: np.ndarray, dataset: str) -> Path:
        """Varianza acumulada de un PCA ajustado solo con TRAIN."""
        n_max = min(X_train.shape[0] - 1, X_train.shape[1], 120)
        modelo = PCA(n_components=n_max, svd_solver="full", random_state=RANDOM_STATE)
        modelo.fit(X_train)
        curva = np.cumsum(modelo.explained_variance_ratio_)
        ejes = np.arange(1, n_max + 1)

        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        ax.plot(ejes, curva, color="#0076BA", linewidth=1.8)
        for d in DIMENSIONES:
            if d <= n_max:
                ax.scatter([d], [curva[d - 1]], color="#B51700", zorder=3)
                ax.annotate(
                    f"d={d}\n{curva[d - 1]:.0%}",
                    (d, curva[d - 1]),
                    textcoords="offset points",
                    xytext=(8, 8),
                    fontsize=8,
                )
        ax.set_xlabel("Componentes conservadas (d)")
        ax.set_ylabel("Varianza acumulada")
        ax.set_ylim(0, 1.02)
        ax.set_title(f"PCA en TRAIN · {dataset}")
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        return self._guardar(fig, f"curva_pca_{dataset}.png")

    def eigenfaces(
        self,
        media: np.ndarray,
        componentes: np.ndarray,
        dataset: str,
        cantidad: int = 11,
    ) -> Path:
        """Rostro medio y las primeras componentes reordenadas a H×W."""
        cantidad = min(cantidad, componentes.shape[0])
        fig, ejes = plt.subplots(3, 4, figsize=(8, 6.2))
        celdas = list(ejes.ravel())
        self._pintar_rostro(celdas[0], media, "media TRAIN")
        for i in range(cantidad):
            self._pintar_rostro(celdas[i + 1], componentes[i], f"PC{i + 1}")
        for ax in celdas[cantidad + 1 :]:
            ax.axis("off")
        fig.suptitle(f"Eigenfaces · {dataset}", fontsize=12)
        fig.tight_layout()
        return self._guardar(fig, f"eigenfaces_{dataset}.png")

    def fisherfaces(
        self,
        patrones: np.ndarray,
        dataset: str,
        d_pca: int,
        d_lda: int,
        cantidad: int = 8,
    ) -> Path:
        """Primeras direcciones discriminantes vistas como imagen.

        Cada patrón se divide por su máximo absoluto. Rojo y azul son pesos
        de signo opuesto, no tonos de piel.
        """
        cantidad = min(cantidad, patrones.shape[0])
        fig, ejes = plt.subplots(2, 4, figsize=(8.4, 4.8), layout="constrained")
        celdas = list(ejes.ravel())
        for i in range(cantidad):
            patron = patrones[i] / np.abs(patrones[i]).max()
            imagen = celdas[i].imshow(patron, cmap="RdBu_r", vmin=-1, vmax=1)
            celdas[i].set_title(f"Fisherface {i + 1}", fontsize=8)
        for ax in celdas:
            ax.axis("off")
        fig.suptitle(f"Fisherfaces · {dataset} · PCA({d_pca}) → LDA({d_lda})", fontsize=12)
        fig.colorbar(imagen, ax=celdas, fraction=0.03, label="Peso relativo del píxel")
        return self._guardar(fig, f"fisherfaces_{dataset}.png")

    def confusion(
        self,
        matriz: np.ndarray,
        dataset: str,
        embedding: str,
        modelo: str,
        d: int,
        normalizar: bool = False,
    ) -> Path:
        """Con ``normalizar`` cada fila suma 1 y la diagonal es el recall por identidad."""
        titulo = f"Confusión · {dataset} · {embedding} d={d} · {modelo}"
        etiqueta = "Fotos de TEST"
        if normalizar:
            matriz = matriz / np.maximum(matriz.sum(axis=1, keepdims=True), 1)
            titulo += "\nnormalizada por identidad real"
            etiqueta = "Fracción de las fotos de la identidad real"
        fig, ax = plt.subplots(figsize=(7.2, 6.2))
        imagen = ax.imshow(matriz, cmap="Blues", vmin=0, vmax=1 if normalizar else None)
        ax.set_xlabel("Identidad predicha")
        ax.set_ylabel("Identidad real")
        ax.set_title(titulo)
        fig.colorbar(imagen, ax=ax, fraction=0.046, label=etiqueta)
        fig.tight_layout()
        nombre = f"confusion_{dataset}_{embedding}_{modelo}_d{d}.png"
        return self._guardar(fig, nombre)

    def mapa(
        self,
        Z: np.ndarray,
        y: np.ndarray,
        metodo: str,
        dataset: str,
        embedding: str,
        d: int,
    ) -> Path:
        """Mapa 2D exploratorio de Z_train. No se usa para calcular accuracy."""
        coordenadas, detalle = self._proyectar_2d(Z, metodo)
        fig, ax = plt.subplots(figsize=(7.2, 5.6))
        dispersion = ax.scatter(
            coordenadas[:, 0],
            coordenadas[:, 1],
            c=y,
            cmap="hsv",
            s=16,
            alpha=0.9,
        )
        ax.set_xlabel("Eje 1")
        ax.set_ylabel("Eje 2")
        ax.set_title(f"{metodo.upper()} · {dataset} · {embedding} d={d}\n{detalle}")
        fig.colorbar(dispersion, ax=ax, fraction=0.046, label="Identidad")
        fig.tight_layout()
        nombre = f"{metodo}_{dataset}_{embedding}_d{d}.png"
        return self._guardar(fig, nombre)

    def _proyectar_2d(self, Z: np.ndarray, metodo: str) -> tuple[np.ndarray, str]:
        n = Z.shape[0]
        if metodo == "tsne":
            perplexity = min(TSNE_PERPLEXITY, max(5.0, (n - 1) / 3))
            modelo = TSNE(
                n_components=2,
                perplexity=perplexity,
                init="pca",
                learning_rate="auto",
                random_state=RANDOM_STATE,
            )
            detalle = f"perplexity={perplexity:.0f}, random_state={RANDOM_STATE}"
            return modelo.fit_transform(Z), detalle
        if metodo == "umap":
            import umap

            n_neighbors = min(UMAP_N_NEIGHBORS, n - 1)
            modelo = umap.UMAP(
                n_components=2,
                n_neighbors=n_neighbors,
                min_dist=UMAP_MIN_DIST,
                random_state=RANDOM_STATE,
                n_jobs=1,
            )
            detalle = (
                f"n_neighbors={n_neighbors}, min_dist={UMAP_MIN_DIST}, "
                f"random_state={RANDOM_STATE}"
            )
            return modelo.fit_transform(Z), detalle
        raise ValueError(f"Método de mapa desconocido: {metodo}")

    @staticmethod
    def _pintar_rostro(ax: plt.Axes, imagen: np.ndarray, titulo: str) -> None:
        ax.imshow(imagen, cmap="gray")
        ax.set_title(titulo, fontsize=8)
        ax.axis("off")

    def _guardar(self, fig: plt.Figure, nombre: str) -> Path:
        ruta = self.directorio / nombre
        fig.savefig(ruta, dpi=120)
        plt.close(fig)
        return ruta
