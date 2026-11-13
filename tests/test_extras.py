"""Análisis extendido: perturbaciones, identidades desconocidas y barrido de d_PCA."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import accuracy_score

from src.datos.division import DivisorIdentidades
from src.embeddings import (
    ExtractorFisherfaces,
    ExtractorLDA,
    ExtractorPCA,
    ExtractorPCASinPrimeras,
)
from src.embeddings.fabrica import FabricaExtractores
from src.evaluacion.metricas import tasa_falsa_aceptacion
from src.evaluacion.perturbaciones import iluminar, ocluir, ventanas
from src.extras import ExperimentosExtra


class FabricaPequena(FabricaExtractores):
    """Embeddings que caben en las fotos sintéticas de 8×8."""

    def para_extras(self, n_clases: int, d: int = 5):
        d_lda = self.dimension_lda(d, n_clases)
        return [ExtractorPCA(d), ExtractorFisherfaces(d_pca=10, d_lda=d_lda)]


def test_iluminar_con_intensidad_cero_no_cambia_la_foto(conjunto):
    assert np.array_equal(iluminar(conjunto.X, conjunto.alto, conjunto.ancho, 0.0), conjunto.X)


def test_iluminar_oscurece_la_izquierda_y_respeta_el_rango():
    X = np.full((3, 4 * 6), 0.5)
    resultado = iluminar(X, alto=4, ancho=6, intensidad=0.8).reshape(3, 4, 6)
    assert resultado.shape == (3, 4, 6)
    assert resultado.min() >= 0.0 and resultado.max() <= 1.0
    assert (resultado[:, :, 0] < 0.5).all() and (resultado[:, :, -1] > 0.5).all()


def test_iluminar_rechaza_intensidad_negativa(conjunto):
    with pytest.raises(ValueError):
        iluminar(conjunto.X, conjunto.alto, conjunto.ancho, -0.1)


def test_ocluir_solo_tapa_el_cuadrado_pedido():
    X = np.ones((2, 8 * 8))
    tapada = ocluir(X, alto=8, ancho=8, fila=2, columna=4, tamano=3).reshape(2, 8, 8)
    assert (tapada[:, 2:5, 4:7] == 0).all()
    assert tapada.sum() == 2 * (64 - 9)


def test_las_ventanas_cubren_toda_la_imagen():
    cubierto = np.zeros((62, 47), dtype=int)
    for fila, columna in ventanas(62, 47, 8):
        cubierto[fila : fila + 8, columna : columna + 8] += 1
    assert (cubierto == 1).all()


def test_pca_sin_primeras_descarta_las_primeras_componentes(particion):
    sin_tres = ExtractorPCASinPrimeras(5, descartar=3).fit(particion.X_train)
    completo = ExtractorPCA(8).fit(particion.X_train)
    assert sin_tres.dimension == 5
    assert sin_tres.nombre == "pca_sin3"
    esperado = completo.transform(particion.X_test)[:, 3:]
    assert np.allclose(sin_tres.transform(particion.X_test), esperado)


def test_tasa_falsa_aceptacion_con_un_ejemplo_a_mano():
    conocidas = np.arange(1, 11, dtype=float)  # 1..10: aceptar el 90% deja el umbral en 1,9
    desconocidas = np.array([0.0, 1.0, 2.0, 5.0])
    assert tasa_falsa_aceptacion(conocidas, desconocidas, aceptacion=0.9) == pytest.approx(0.5)


def test_las_identidades_desconocidas_nunca_entran_a_train(conjunto):
    particion = DivisorIdentidades(fraccion_conocidas=0.5, random_state=0).dividir(conjunto)
    desconocidas = np.setdiff1d(np.unique(conjunto.y), particion.conocidas)
    assert particion.conocidas.size == 3 and desconocidas.size == 3
    assert set(np.unique(particion.y_train)) == set(particion.conocidas)
    assert set(np.unique(particion.y_test)) <= set(particion.conocidas)
    assert set(np.unique(particion.y_desconocidas)) == set(desconocidas)
    total = particion.y_train.size + particion.y_test.size + particion.y_desconocidas.size
    assert total == conjunto.y.size


def test_la_luz_solo_cambia_test(conjunto, particion):
    X_train, X_test = particion.X_train.copy(), particion.X_test.copy()
    extras = ExperimentosExtra(fabrica_extractores=FabricaPequena())
    modelos = extras.ajustar(particion.X_train, particion.y_train, conjunto.n_clases)
    filas = extras._iluminacion(conjunto, particion, modelos, semilla=42)

    extractor, clasificadores, _ = modelos["pca"]
    limpia = clasificadores["svm"].predict(extractor.transform(particion.X_test))
    sin_luz = next(
        f
        for f in filas
        if f["intensidad"] == 0 and f["embedding"] == "pca" and f["modelo"] == "svm"
    )
    assert sin_luz["accuracy"] == accuracy_score(particion.y_test, limpia)
    assert np.array_equal(particion.X_train, X_train)
    assert np.array_equal(particion.X_test, X_test)


def test_desconocidos_entrega_auc_y_falsa_aceptacion_validas(conjunto):
    extras = ExperimentosExtra(fabrica_extractores=FabricaPequena())
    filas, curvas = extras._desconocidos(conjunto, semilla=0)
    assert {f["puntaje"] for f in filas} == {"distancia", "probabilidad"}
    assert len(filas) == 2 * len(curvas)
    assert all(0 <= f["auc"] <= 1 and 0 <= f["falsa_aceptacion"] <= 1 for f in filas)


def test_recortar_un_pca_grande_equivale_a_fisherfaces(particion):
    grande = ExtractorPCA(40).fit(particion.X_train)
    recortado = grande.transform(particion.X_train)[:, :10]
    lda = ExtractorLDA(5).fit(recortado, particion.y_train)
    fisher = ExtractorFisherfaces(d_pca=10, d_lda=5).fit(particion.X_train, particion.y_train)
    esperado = fisher.transform(particion.X_test)
    assert np.allclose(lda.transform(grande.transform(particion.X_test)[:, :10]), esperado)


def test_el_barrido_de_dpca_termina_en_n_menos_c():
    valores = ExperimentosExtra.valores_d_pca(n_train=300, n_clases=40, d_lda=30)
    assert valores[-1] == 260
    assert valores == sorted(valores)
    assert all(30 <= d <= 260 for d in valores)
