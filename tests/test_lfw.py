"""Selección de identidades de LFW y contrato del conjunto real."""

from __future__ import annotations

import numpy as np
import pytest

from src.config import DIR_CACHE
from src.datos.lfw import CargadorLFW


def test_la_seleccion_conserva_las_mas_frecuentes_y_reindexa():
    # Fotos por identidad: Ana 2, Beto 4, Caro 1, Dani 4, Eli 3.
    nombres = ["Ana", "Beto", "Caro", "Dani", "Eli"]
    y = np.array([0, 1, 1, 3, 4, 1, 3, 0, 4, 3, 2, 1, 4, 3])
    mascara, y_nuevo, elegidos = CargadorLFW.seleccionar_identidades(y, nombres, maximo=3)

    assert elegidos == ["Beto", "Dani", "Eli"]
    assert mascara.sum() == 11
    assert np.array_equal(np.unique(y_nuevo), [0, 1, 2])
    # Cada foto sigue apuntando a la misma persona después de reindexar.
    assert [elegidos[i] for i in y_nuevo] == [nombres[i] for i in y[mascara]]


def test_el_empate_en_el_corte_se_resuelve_por_orden_alfabetico():
    nombres = ["Ana", "Beto", "Caro"]
    y = np.array([0, 0, 1, 1, 2, 2, 2])
    _, _, elegidos = CargadorLFW.seleccionar_identidades(y, nombres, maximo=2)
    assert elegidos == ["Ana", "Caro"]


@pytest.mark.skipif(
    not (DIR_CACHE / "lfw_home").exists(),
    reason="LFW no está descargado en la caché",
)
def test_lfw_real_cumple_el_contrato():
    conjunto = CargadorLFW().cargar()
    conteos = np.bincount(conjunto.y)

    assert conjunto.n_clases == 50
    assert np.array_equal(np.unique(conjunto.y), np.arange(50))
    assert conteos.min() >= 20
    assert conjunto.X.shape == (conjunto.y.size, conjunto.alto * conjunto.ancho)
    assert conjunto.images.shape == (conjunto.y.size, conjunto.alto, conjunto.ancho)
    assert len(set(conjunto.nombres_identidades)) == 50
    assert 0.0 <= conjunto.X.min() and conjunto.X.max() <= 1.0
    # La identidad con más fotos de LFW conserva su nombre tras el reindexado.
    assert conjunto.nombres_identidades[int(conteos.argmax())] == "George W Bush"
