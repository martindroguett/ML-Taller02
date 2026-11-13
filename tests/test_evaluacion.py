"""Métricas, medición de tiempo y reproducibilidad con semilla fija."""

from __future__ import annotations

import numpy as np
import pytest

from src.clasificadores import FabricaClasificadores
from src.datos.division import DivisorEstratificado
from src.embeddings import ExtractorFisherfaces
from src.evaluacion.metricas import Evaluador, cronometrar


def _evaluar(conjunto, clasificador: str):
    particion = DivisorEstratificado(test_size=0.25, random_state=42).dividir(conjunto)
    fisher = ExtractorFisherfaces(d_pca=10, d_lda=3).fit(particion.X_train, particion.y_train)
    medicion = Evaluador(repeticiones=1).evaluar(
        FabricaClasificadores().crear(clasificador),
        fisher.transform(particion.X_train),
        particion.y_train,
        fisher.transform(particion.X_test),
        particion.y_test,
    )
    return particion, medicion


def test_las_metricas_salen_de_las_predicciones(conjunto):
    particion, medicion = _evaluar(conjunto, "knn")
    aciertos = medicion.predicciones == particion.y_test
    assert medicion.accuracy == pytest.approx(aciertos.mean())
    assert medicion.matriz.sum() == particion.y_test.size
    assert np.trace(medicion.matriz) == aciertos.sum()
    assert 0.0 <= medicion.macro_f1 <= 1.0


@pytest.mark.parametrize("clasificador", ["svm", "knn", "arbol"])
def test_dos_ejecuciones_con_la_misma_semilla_coinciden(conjunto, clasificador):
    _, primera = _evaluar(conjunto, clasificador)
    _, segunda = _evaluar(conjunto, clasificador)
    assert np.array_equal(primera.predicciones, segunda.predicciones)
    assert primera.macro_f1 == segunda.macro_f1


def test_cronometrar_repite_la_accion():
    llamadas = []
    resultado, tiempo = cronometrar(lambda: llamadas.append(1) or len(llamadas), repeticiones=3)
    assert resultado == 3
    assert tiempo >= 0


def test_cronometrar_exige_al_menos_una_repeticion():
    with pytest.raises(ValueError):
        cronometrar(lambda: None, repeticiones=0)
