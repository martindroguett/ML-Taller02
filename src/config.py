"""Rutas y constantes compartidas del laboratorio.

Toda ruta se calcula desde la raíz del repositorio para que el pipeline
funcione igual si se ejecuta `python main.py` desde cualquier cwd.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

RAIZ = Path(__file__).resolve().parent.parent
load_dotenv(RAIZ / ".env")

DATA_DIR = RAIZ / "data"
DIR_RESULTADOS = DATA_DIR / "resultados"
DIR_MODELOS = DATA_DIR / "modelos"
DIR_FIGURAS = RAIZ / "figuras"

_cache = os.getenv("SKLEARN_DATA_HOME", "data/cache").strip() or "data/cache"
DIR_CACHE = Path(_cache) if Path(_cache).is_absolute() else RAIZ / _cache

RANDOM_STATE = int(os.getenv("RANDOM_STATE", "42"))
TEST_SIZE = float(os.getenv("TEST_SIZE", "0.25"))

DIMENSIONES = (10, 20, 30)
DIMENSION_MAPAS = 20

TSNE_PERPLEXITY = 30
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.1

CLASIFICADORES = ("svm", "knn", "arbol")
EMBEDDINGS = ("pca", "lda", "fisherfaces")

LFW_MIN_FOTOS = 20
LFW_MAX_IDENTIDADES = 50
LFW_RESIZE = 0.5

REPETICIONES_TIEMPO = 5

SEMILLAS_ROBUSTEZ = tuple(RANDOM_STATE + i for i in range(5))
D_PCA_VARIANTE = 100

# Extras
D_EXTRAS = 30
CLASIFICADORES_EXTRAS = ("svm", "knn")
INTENSIDADES_LUZ = (0.0, 0.2, 0.4, 0.6, 0.8)
VENTANA_OCLUSION = 8
FRACCION_CONOCIDAS = 0.75
ACEPTACION_CONOCIDOS = 0.9
BARRIDO_D_PCA = (30, 50, 100, 200, 500, 1000)


def ruta_modelo(dataset: str, embedding: str, modelo: str, d: int) -> Path:
    """Artefacto joblib de una combinación ya ajustada en TRAIN."""
    return DIR_MODELOS / f"{dataset}_{embedding}_{modelo}_d{d}.joblib"


def asegurar_directorios() -> None:
    """Crea caché, modelos, resultados y figuras si todavía no existen."""
    for ruta in (DIR_CACHE, DIR_MODELOS, DIR_RESULTADOS, DIR_FIGURAS):
        ruta.mkdir(parents=True, exist_ok=True)
