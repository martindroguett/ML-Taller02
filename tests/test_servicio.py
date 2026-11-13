"""La interfaz y la tabla de resultados deben decidir la misma identidad."""

from __future__ import annotations

import pytest

from src.interfaz import servicio as modulo_servicio


@pytest.mark.parametrize("embedding", ["pca", "fisherfaces"])
def test_la_interfaz_predice_lo_mismo_que_predict(embedding, monkeypatch, tmp_path):
    # Los modelos de este test van a una carpeta temporal, no a data/modelos.
    monkeypatch.setattr(
        modulo_servicio,
        "ruta_modelo",
        lambda *partes: tmp_path / ("_".join(str(p) for p in partes) + ".joblib"),
    )
    servicio = modulo_servicio.ServicioReconocimiento()
    bundle = servicio._obtener_bundle(embedding, "svm", 10)
    Z_test = bundle["extractor"].transform(servicio.particion.X_test)
    esperadas = bundle["clasificador"].predict(Z_test)

    for indice, esperada in enumerate(esperadas):
        prediccion = servicio.predecir(indice, embedding, "svm", 10)
        assert prediccion.identidad_predicha == esperada
        assert prediccion.acierto == (esperada == servicio.particion.y_test[indice])
