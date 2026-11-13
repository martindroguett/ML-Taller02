#!/usr/bin/env python3
"""Interfaz web para revisar rostros de prueba con un modelo ya ajustado.

Uso (con el entorno conda activado):

    streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import streamlit as st

from src.config import DIMENSIONES, EMBEDDINGS
from src.interfaz.servicio import ServicioReconocimiento


st.set_page_config(page_title="Embeddings faciales", layout="wide")


@st.cache_resource
def obtener_servicio() -> ServicioReconocimiento:
    return ServicioReconocimiento()


def main() -> None:
    servicio = obtener_servicio()
    st.title("Reconocimiento de rostros con embeddings lineales")
    st.caption(
        "Cada fotografía pertenece al conjunto de prueba de Olivetti "
        f"({servicio.particion.images_test.shape[0]} imágenes). "
        "No participó del ajuste de PCA, LDA, Fisherfaces ni del clasificador."
    )

    with st.sidebar:
        st.header("Modelo")
        embedding = st.radio(
            "Representación",
            options=list(EMBEDDINGS),
            format_func=_etiqueta_embedding,
        )
        clasificador = st.radio(
            "Clasificador",
            options=["svm", "knn", "arbol"],
            format_func=_etiqueta_clasificador,
        )
        d = st.select_slider("Componentes (d)", options=list(DIMENSIONES))
        if embedding == "fisherfaces":
            st.caption(f"Fisherfaces: PCA conserva {d} componentes y LDA proyecta a {d}.")
        etiquetas = servicio.etiquetas_prueba()
        indice = st.selectbox(
            "Fotografía de prueba",
            options=range(len(etiquetas)),
            format_func=lambda i: etiquetas[i],
        )

    with st.spinner("Cargando el modelo ajustado en TRAIN..."):
        prediccion = servicio.predecir(indice, embedding, clasificador, int(d))

    columna_foto, columna_resultado = st.columns([1, 1.4], gap="large")
    with columna_foto:
        st.image(
            servicio.imagen_prueba(indice),
            caption=prediccion.nombre_real,
            clamp=True,
            width=280,
        )
    with columna_resultado:
        if prediccion.acierto:
            st.success("El modelo identificó la cara.")
        else:
            st.error("El modelo no identificó la cara.")
        real, predicha, probabilidad = st.columns(3)
        real.markdown(f"**Identidad real**\n\n{prediccion.nombre_real}")
        predicha.markdown(f"**Identidad predicha**\n\n{prediccion.nombre_predicho}")
        probabilidad.markdown(f"**Probabilidad**\n\n{prediccion.probabilidad:.1%}")
        st.caption(prediccion.nota_probabilidad)
        st.subheader("Cinco identidades más probables")
        tabla = pd.DataFrame(prediccion.top, columns=["identidad", "probabilidad"])
        st.bar_chart(
            tabla,
            x="identidad",
            y="probabilidad",
            horizontal=True,
            sort=False,
        )


def _etiqueta_embedding(nombre: str) -> str:
    return {"pca": "PCA", "lda": "LDA", "fisherfaces": "Fisherfaces"}[nombre]


def _etiqueta_clasificador(nombre: str) -> str:
    return {"svm": "SVM", "knn": "k-NN", "arbol": "Árbol"}[nombre]


main()
