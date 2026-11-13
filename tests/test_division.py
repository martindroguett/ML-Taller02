"""El split deja TRAIN y TEST disjuntos, alineados y reproducibles."""

from __future__ import annotations

import numpy as np

from src.datos.division import DivisorEstratificado


def test_train_y_test_son_disjuntos_y_cubren_todo(conjunto, particion):
    juntos = np.concatenate([particion.indices_train, particion.indices_test])
    assert np.array_equal(np.sort(juntos), np.arange(conjunto.X.shape[0]))


def test_formas_alineadas(conjunto, particion):
    n_test = particion.y_test.size
    n_train = conjunto.X.shape[0] - n_test
    assert particion.X_train.shape == (n_train, conjunto.alto * conjunto.ancho)
    assert particion.X_test.shape == (n_test, conjunto.alto * conjunto.ancho)
    assert particion.images_test.shape == (n_test, conjunto.alto, conjunto.ancho)
    assert np.array_equal(particion.X_test, conjunto.X[particion.indices_test])
    assert np.array_equal(particion.y_test, conjunto.y[particion.indices_test])


def test_estratificado_mantiene_la_proporcion(conjunto, particion):
    # 20 fotos por identidad y test_size=0.25: 5 van a TEST y 15 a TRAIN.
    assert np.bincount(particion.y_test).tolist() == [5] * conjunto.n_clases
    assert np.bincount(particion.y_train).tolist() == [15] * conjunto.n_clases


def test_la_misma_semilla_repite_la_particion(conjunto):
    primera = DivisorEstratificado(random_state=42).dividir(conjunto)
    segunda = DivisorEstratificado(random_state=42).dividir(conjunto)
    otra = DivisorEstratificado(random_state=7).dividir(conjunto)
    assert np.array_equal(primera.indices_test, segunda.indices_test)
    assert not np.array_equal(primera.indices_test, otra.indices_test)
