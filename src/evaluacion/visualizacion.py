"""Figuras del experimento: curva PCA, eigenfaces, fisherfaces, confusión, mapas 2D y extras."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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

    def iluminacion(self, tabla: pd.DataFrame, dataset: str) -> Path:
        """Accuracy en TEST según la intensidad de la luz, media ± desviación entre semillas."""
        modelos = list(tabla["modelo"].unique())
        fig, ejes = plt.subplots(
            1, len(modelos), figsize=(5.4 * len(modelos), 4.4), sharey=True, squeeze=False
        )
        for ax, modelo in zip(ejes[0], modelos):
            resumen = (
                tabla[tabla["modelo"] == modelo]
                .groupby(["embedding", "intensidad"], sort=False)["accuracy"]
                .agg(["mean", "std"])
                .reset_index()
            )
            for embedding, curva in resumen.groupby("embedding", sort=False):
                ax.plot(curva["intensidad"], curva["mean"], marker="o", label=embedding)
                ax.fill_between(
                    curva["intensidad"],
                    curva["mean"] - curva["std"],
                    curva["mean"] + curva["std"],
                    alpha=0.15,
                )
            ax.set_title(modelo)
            ax.set_xlabel("Intensidad de la luz lateral")
            ax.grid(True, alpha=0.3)
        ejes[0][0].set_ylabel("Accuracy en TEST")
        ejes[0][0].set_ylim(0, 1.02)
        ejes[0][-1].legend(fontsize=8)
        fig.suptitle(f"Iluminación simulada en TEST · {dataset}", fontsize=12)
        fig.tight_layout()
        return self._guardar(fig, f"extras_iluminacion_{dataset}.png")

    def oclusion(self, caidas: dict[str, np.ndarray], media: np.ndarray, dataset: str) -> Path:
        """Caída de accuracy al tapar cada cuadrado, dibujada sobre el rostro medio de TRAIN."""
        tope = max(float(caida.max()) for caida in caidas.values())
        fig, ejes = plt.subplots(
            1, len(caidas), figsize=(2.7 * len(caidas), 3.6), layout="constrained", squeeze=False
        )
        for ax, (embedding, caida) in zip(ejes[0], caidas.items()):
            ax.imshow(media, cmap="gray")
            imagen = ax.imshow(caida, cmap="Reds", alpha=0.65, vmin=0, vmax=tope)
            ax.set_title(embedding, fontsize=9)
            ax.axis("off")
        fig.colorbar(imagen, ax=list(ejes[0]), fraction=0.03, label="Caída de accuracy (SVM)")
        fig.suptitle(f"Importancia por oclusión · {dataset}", fontsize=12)
        return self._guardar(fig, f"extras_oclusion_{dataset}.png")

    def desconocidos(
        self,
        curvas: dict[str, tuple[np.ndarray, np.ndarray, float]],
        dataset: str,
    ) -> Path:
        """Curva ROC de conocidos contra desconocidos con la distancia al vecino más cercano."""
        fig, ax = plt.subplots(figsize=(5.8, 5.2))
        for embedding, (falsos, verdaderos, auc) in curvas.items():
            ax.plot(falsos, verdaderos, label=f"{embedding} (AUC={auc:.2f})")
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
        ax.set_xlabel("Desconocidos aceptados")
        ax.set_ylabel("Conocidos aceptados")
        ax.set_title(
            f"Identidades desconocidas · {dataset}\ndistancia al vecino más cercano de TRAIN"
        )
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return self._guardar(fig, f"extras_desconocidos_{dataset}.png")

    def barrido_dpca(self, tabla: pd.DataFrame, dataset: str) -> Path:
        """Accuracy de Fisherfaces en TRAIN y TEST según d_PCA, media ± desviación."""
        resumen = tabla.groupby("d_pca")[["accuracy_train", "accuracy_test"]].agg(["mean", "std"])
        fig, ax = plt.subplots(figsize=(7.2, 4.4))
        for columna, etiqueta in (("accuracy_train", "TRAIN"), ("accuracy_test", "TEST")):
            media = resumen[(columna, "mean")]
            desviacion = resumen[(columna, "std")]
            ax.plot(resumen.index, media, marker="o", label=etiqueta)
            ax.fill_between(resumen.index, media - desviacion, media + desviacion, alpha=0.15)
        maximo = int(resumen.index.max())
        ax.annotate(
            "N − C",
            (maximo, resumen[("accuracy_test", "mean")].iloc[-1]),
            textcoords="offset points",
            xytext=(-30, 10),
            fontsize=8,
        )
        ax.set_xscale("log")
        ax.set_xticks(resumen.index, [str(int(d)) for d in resumen.index])
        ax.minorticks_off()
        ax.set_xlabel("d_PCA: componentes que recibe LDA (escala log)")
        ax.set_ylabel("Accuracy (SVM)")
        ax.set_ylim(0, 1.02)
        d_lda = int(tabla["d_lda"].iloc[0])
        ax.set_title(f"Fisherfaces según d_PCA · {dataset} · d_LDA={d_lda}")
        ax.legend(loc="lower left")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return self._guardar(fig, f"extras_barrido_dpca_{dataset}.png")



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
