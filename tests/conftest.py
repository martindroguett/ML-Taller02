"""Datos sintéticos compartidos por los tests."""

from __future__ import annotations

import numpy as np
import pytest

from src.datos.division import DivisorEstratificado
from src.modelos import ConjuntoRostros, Particion


@pytest.fixture
def conjunto() -> ConjuntoRostros:
    """Seis identidades con veinte fotos de 8×8: la media de la identidad más ruido."""
    rng = np.random.default_rng(0)
    n_clases, por_clase, alto, ancho = 6, 20, 8, 8
    medias = rng.normal(size=(n_clases, alto * ancho))
    y = np.repeat(np.arange(n_clases), por_clase)
    X = medias[y] + 0.3 * rng.normal(size=(y.size, alto * ancho))
    return ConjuntoRostros(
        nombre="sintetico",
        images=X.reshape(-1, alto, ancho),
        X=X,
        y=y,
        nombres_identidades=[f"Identidad {i}" for i in range(n_clases)],
        alto=alto,
        ancho=ancho,
    )


@pytest.fixture
def particion(conjunto: ConjuntoRostros) -> Particion:
    return DivisorEstratificado(test_size=0.25, random_state=42).dividir(conjunto)
