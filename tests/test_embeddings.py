"""PCA, LDA y Fisherfaces: formas, restricciones de dimensión y ajuste solo con TRAIN."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

from src.embeddings import ExtractorFisherfaces, ExtractorLDA, ExtractorPCA, FabricaExtractores
from src.pipeline import PipelineExperimentos


@pytest.mark.parametrize(
    "extractor",
    [ExtractorPCA(10), ExtractorLDA(3), ExtractorFisherfaces(d_pca=10, d_lda=3)],
    ids=["pca", "lda", "fisherfaces"],
)
def test_el_embedding_tiene_una_fila_por_foto_y_d_columnas(extractor, particion):
    extractor.fit(particion.X_train, particion.y_train)
    d = extractor.dimension
    assert extractor.transform(particion.X_train).shape == (particion.y_train.size, d)
    assert extractor.transform(particion.X_test).shape == (particion.y_test.size, d)


def test_pca_centra_con_la_media_de_train(conjunto, particion):
    pca = ExtractorPCA(10).fit(particion.X_train)
    media = pca.rostro_medio(conjunto.alto, conjunto.ancho).ravel()
    assert np.allclose(media, particion.X_train.mean(axis=0))
    assert not np.allclose(media, conjunto.X.mean(axis=0))


def test_fisherfaces_es_pca_y_despues_lda_ajustados_en_train(particion):
    fisher = ExtractorFisherfaces(d_pca=10, d_lda=3).fit(particion.X_train, particion.y_train)
    pca = ExtractorPCA(10).fit(particion.X_train)
    lda = ExtractorLDA(3).fit(pca.transform(particion.X_train), particion.y_train)
    esperado = lda.transform(pca.transform(particion.X_test))
    assert np.allclose(fisher.transform(particion.X_test), esperado)


def test_transformar_test_no_reajusta(particion):
    fisher = ExtractorFisherfaces(d_pca=10, d_lda=3).fit(particion.X_train, particion.y_train)
    antes = fisher.transform(particion.X_train)
    fisher.transform(particion.X_test)
    assert np.array_equal(antes, fisher.transform(particion.X_train))


class ExtractorEspia(ExtractorPCA):
    """Recuerda con qué matriz se llamó a fit."""

    def fit(self, X, y=None):
        self.visto_en_fit = X
        return super().fit(X, y)


def test_el_pipeline_ajusta_el_embedding_solo_con_train(particion):
    espia = ExtractorEspia(5)
    Z_train, Z_test, tiempo = PipelineExperimentos()._proyectar(espia, particion)
    assert espia.visto_en_fit is particion.X_train
    assert Z_train.shape == (particion.y_train.size, 5)
    assert Z_test.shape == (particion.y_test.size, 5)
    assert tiempo > 0


def test_fisherfaces_rechaza_dimensiones_invalidas(particion):
    n_train = particion.y_train.size
    n_clases = np.unique(particion.y_train).size
    with pytest.raises(ValueError, match="d_pca"):
        ExtractorFisherfaces(d_pca=3, d_lda=4)
    with pytest.raises(ValueError, match="C-1"):
        ExtractorFisherfaces(d_pca=10, d_lda=n_clases).fit(particion.X_train, particion.y_train)
    with pytest.raises(ValueError, match="N-C"):
        demasiadas = n_train - n_clases + 1
        ExtractorFisherfaces(d_pca=demasiadas, d_lda=3).fit(particion.X_train, particion.y_train)


def test_lda_rechaza_mas_de_c_menos_1_direcciones(particion):
    n_clases = np.unique(particion.y_train).size
    with pytest.raises(ValueError, match="C-1"):
        ExtractorLDA(n_clases).fit(particion.X_train, particion.y_train)


def test_transform_antes_de_fit_falla(particion):
    with pytest.raises(NotFittedError):
        ExtractorFisherfaces(d_pca=10, d_lda=3).transform(particion.X_test)


def test_las_fisherfaces_son_las_direcciones_de_la_proyeccion(conjunto, particion):
    fisher = ExtractorFisherfaces(d_pca=10, d_lda=3).fit(particion.X_train, particion.y_train)
    patrones = fisher.componentes_imagen(conjunto.alto, conjunto.ancho)
    assert patrones.shape == (3, conjunto.alto, conjunto.ancho)
    # z = Wᵀ(x − μ): proyectar con los patrones difiere de transform en una constante.
    diferencia = fisher.transform(particion.X_test) - particion.X_test @ patrones.reshape(3, -1).T
    assert np.allclose(diferencia, diferencia[0])


def test_la_fabrica_acota_lda_a_c_menos_1():
    fabrica = FabricaExtractores()
    assert fabrica.dimension_lda(30, n_clases=40) == 30
    assert fabrica.dimension_lda(30, n_clases=6) == 5
    extractores = fabrica.para_experimento(n_clases=40)
    assert {e.nombre for e in extractores} == {"pca", "lda", "fisherfaces"}
    assert all(e.dimension <= 39 for e in extractores)
