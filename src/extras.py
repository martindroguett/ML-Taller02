"""Análisis extendido: luz, oclusión, identidades desconocidas y d_PCA de Fisherfaces.

Los modelos se ajustan como en PipelineExperimentos, solo con TRAIN, y las perturbaciones tocan solo TEST.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve
from sklearn.neighbors import NearestNeighbors

from src.clasificadores.fabrica import FabricaClasificadores
from src.config import (
    BARRIDO_D_PCA,
    CLASIFICADORES_EXTRAS,
    D_EXTRAS,
    DIR_RESULTADOS,
    INTENSIDADES_LUZ,
    SEMILLAS_ROBUSTEZ,
    VENTANA_OCLUSION,
    asegurar_directorios,
)
from src.datos.division import DivisorEstratificado, DivisorIdentidades
from src.datos.lfw import CargadorLFW
from src.datos.olivetti import CargadorOlivetti
from src.embeddings.fabrica import FabricaExtractores
from src.embeddings.lda import ExtractorLDA
from src.embeddings.pca import ExtractorPCA
from src.evaluacion.metricas import tasa_falsa_aceptacion
from src.evaluacion.perturbaciones import iluminar, ocluir, ventanas
from src.evaluacion.visualizacion import Visualizador
from src.modelos import ConjuntoRostros, Particion

TABLAS = ("iluminacion", "oclusion", "desconocidos", "barrido_dpca")


class ExperimentosExtra:
    """Cuatro preguntas que el protocolo no responde, repetidas con varias semillas."""

    def __init__(
        self,
        fabrica_extractores: FabricaExtractores | None = None,
        fabrica_clasificadores: FabricaClasificadores | None = None,
        visualizador: Visualizador | None = None,
        semillas: tuple[int, ...] = SEMILLAS_ROBUSTEZ,
    ) -> None:
        self.fabrica_extractores = fabrica_extractores or FabricaExtractores()
        self.fabrica_clasificadores = fabrica_clasificadores or FabricaClasificadores()
        self.visualizador = visualizador or Visualizador()
        self.semillas = semillas

    def ejecutar(self) -> None:
        asegurar_directorios()
        tablas: dict[str, list[dict]] = {nombre: [] for nombre in TABLAS}
        for cargador in (CargadorOlivetti(), CargadorLFW()):
            conjunto = cargador.cargar()
            print(f"== Extras: {conjunto.nombre} ==")
            curvas_roc: dict = {}
            for semilla in self.semillas:
                particion = DivisorEstratificado(random_state=semilla).dividir(conjunto)
                modelos = self.ajustar(particion.X_train, particion.y_train, conjunto.n_clases)
                tablas["iluminacion"] += self._iluminacion(conjunto, particion, modelos, semilla)
                tablas["oclusion"] += self._oclusion(conjunto, particion, modelos, semilla)
                filas, curvas = self._desconocidos(conjunto, semilla)
                tablas["desconocidos"] += filas
                curvas_roc = curvas_roc or curvas
                tablas["barrido_dpca"] += self._barrido_dpca(conjunto, particion, semilla)
                print(f"  semilla {semilla} lista")
            self._figuras(conjunto, tablas, curvas_roc)
        self._guardar(tablas)

    def ajustar(self, X_train: np.ndarray, y_train: np.ndarray, n_clases: int) -> dict:
        """Ajusta cada embedding y cada clasificador una sola vez, solo con TRAIN."""
        modelos = {}
        for extractor in self.fabrica_extractores.para_extras(n_clases):
            extractor.fit(X_train, y_train)
            Z_train = extractor.transform(X_train)
            clasificadores = {
                nombre: self.fabrica_clasificadores.crear(nombre).fit(Z_train, y_train)
                for nombre in CLASIFICADORES_EXTRAS
            }
            modelos[extractor.nombre] = (extractor, clasificadores, Z_train)
        return modelos

    def _iluminacion(
        self,
        conjunto: ConjuntoRostros,
        particion: Particion,
        modelos: dict,
        semilla: int,
    ) -> list[dict]:
        filas = []
        for intensidad in INTENSIDADES_LUZ:
            X = iluminar(particion.X_test, conjunto.alto, conjunto.ancho, intensidad)
            for embedding, (extractor, clasificadores, _) in modelos.items():
                Z = extractor.transform(X)
                for modelo, clasificador in clasificadores.items():
                    predicciones = clasificador.predict(Z)
                    filas.append(
                        {
                            "dataset": conjunto.nombre,
                            "semilla": semilla,
                            "embedding": embedding,
                            "d": extractor.dimension,
                            "modelo": modelo,
                            "intensidad": intensidad,
                            "accuracy": accuracy_score(particion.y_test, predicciones),
                            "macro_f1": f1_score(
                                particion.y_test, predicciones, average="macro", zero_division=0
                            ),
                        }
                    )
        return filas

    def _oclusion(
        self,
        conjunto: ConjuntoRostros,
        particion: Particion,
        modelos: dict,
        semilla: int,
    ) -> list[dict]:
        """Tapa un cuadrado a la vez y mide cuánto cae la accuracy del SVM."""
        filas = []
        for embedding, (extractor, clasificadores, _) in modelos.items():
            svm = clasificadores["svm"]
            Z_test = extractor.transform(particion.X_test)
            base = accuracy_score(particion.y_test, svm.predict(Z_test))
            for fila, columna in ventanas(conjunto.alto, conjunto.ancho, VENTANA_OCLUSION):
                X = ocluir(
                    particion.X_test, conjunto.alto, conjunto.ancho, fila, columna, VENTANA_OCLUSION
                )
                accuracy = accuracy_score(particion.y_test, svm.predict(extractor.transform(X)))
                filas.append(
                    {
                        "dataset": conjunto.nombre,
                        "semilla": semilla,
                        "embedding": embedding,
                        "fila": fila,
                        "columna": columna,
                        "accuracy": accuracy,
                        "caida": base - accuracy,
                    }
                )
        return filas

    def _desconocidos(self, conjunto: ConjuntoRostros, semilla: int) -> tuple[list[dict], dict]:
        """Entrena con parte de las identidades y mide si rechaza a las que nunca vio."""
        particion = DivisorIdentidades(random_state=semilla).dividir(conjunto)
        modelos = self.ajustar(particion.X_train, particion.y_train, particion.conocidas.size)
        filas, curvas = [], {}
        for embedding, (extractor, clasificadores, Z_train) in modelos.items():
            Z_test = extractor.transform(particion.X_test)
            Z_desconocidas = extractor.transform(particion.X_desconocidas)
            vecinos = NearestNeighbors(n_neighbors=1).fit(Z_train)
            svm = clasificadores["svm"]
            # Un puntaje alto significa "se parece a alguien conocido".
            puntajes = {
                "distancia": (
                    -vecinos.kneighbors(Z_test)[0][:, 0],
                    -vecinos.kneighbors(Z_desconocidas)[0][:, 0],
                ),
                "probabilidad": (
                    svm.predict_proba(Z_test).max(axis=1),
                    svm.predict_proba(Z_desconocidas).max(axis=1),
                ),
            }
            accuracy = accuracy_score(particion.y_test, svm.predict(Z_test))
            for puntaje, (conocidas, desconocidas) in puntajes.items():
                etiquetas = np.r_[np.ones(conocidas.size), np.zeros(desconocidas.size)]
                valores = np.r_[conocidas, desconocidas]
                auc = roc_auc_score(etiquetas, valores)
                filas.append(
                    {
                        "dataset": conjunto.nombre,
                        "semilla": semilla,
                        "embedding": embedding,
                        "puntaje": puntaje,
                        "conocidas": particion.conocidas.size,
                        "auc": auc,
                        "falsa_aceptacion": tasa_falsa_aceptacion(conocidas, desconocidas),
                        "accuracy_conocidas": accuracy,
                    }
                )
                if puntaje == "distancia":
                    falsos, verdaderos, _ = roc_curve(etiquetas, valores)
                    curvas[embedding] = (falsos, verdaderos, auc)
        return filas, curvas

    def _barrido_dpca(
        self,
        conjunto: ConjuntoRostros,
        particion: Particion,
        semilla: int,
    ) -> list[dict]:
        """Fisherfaces con d_LDA fijo y cada vez más componentes PCA antes de LDA.

        PCA se ajusta una sola vez con N − C componentes. Sus primeras d_PCA
        son las mismas que daría PCA(d_PCA), así que no hace falta repetir la
        descomposición para cada valor del barrido.
        """
        d_lda = self.fabrica_extractores.dimension_lda(D_EXTRAS, conjunto.n_clases)
        valores = self.valores_d_pca(particion.y_train.size, conjunto.n_clases, d_lda)
        pca = ExtractorPCA(n_components=valores[-1]).fit(particion.X_train)
        P_train = pca.transform(particion.X_train)
        P_test = pca.transform(particion.X_test)
        filas = []
        for d_pca in valores:
            lda = ExtractorLDA(n_components=d_lda).fit(P_train[:, :d_pca], particion.y_train)
            Z_train = lda.transform(P_train[:, :d_pca])
            svm = self.fabrica_clasificadores.crear("svm").fit(Z_train, particion.y_train)
            predicciones = svm.predict(lda.transform(P_test[:, :d_pca]))
            filas.append(
                {
                    "dataset": conjunto.nombre,
                    "semilla": semilla,
                    "d_pca": d_pca,
                    "d_lda": d_lda,
                    "accuracy_train": accuracy_score(particion.y_train, svm.predict(Z_train)),
                    "accuracy_test": accuracy_score(particion.y_test, predicciones),
                    "macro_f1_test": f1_score(
                        particion.y_test, predicciones, average="macro", zero_division=0
                    ),
                }
            )
        return filas

    @staticmethod
    def valores_d_pca(n_train: int, n_clases: int, d_lda: int) -> list[int]:
        """Valores de BARRIDO_D_PCA entre d_LDA y N − C, más el propio N − C."""
        maximo = n_train - n_clases
        return [d for d in BARRIDO_D_PCA if d_lda <= d < maximo] + [maximo]

    def _figuras(self, conjunto: ConjuntoRostros, tablas: dict, curvas_roc: dict) -> None:
        nombre = conjunto.nombre
        luz = pd.DataFrame(tablas["iluminacion"])
        self.visualizador.iluminacion(luz[luz["dataset"] == nombre], nombre)

        oclusion = pd.DataFrame(tablas["oclusion"])
        oclusion = oclusion[oclusion["dataset"] == nombre]
        referencia = DivisorEstratificado(random_state=self.semillas[0]).dividir(conjunto)
        media = referencia.X_train.mean(axis=0).reshape(conjunto.alto, conjunto.ancho)
        self.visualizador.oclusion(self._mapas_oclusion(oclusion, conjunto), media, nombre)

        self.visualizador.desconocidos(curvas_roc, nombre)

        barrido = pd.DataFrame(tablas["barrido_dpca"])
        self.visualizador.barrido_dpca(barrido[barrido["dataset"] == nombre], nombre)

    @staticmethod
    def _mapas_oclusion(tabla: pd.DataFrame, conjunto: ConjuntoRostros) -> dict[str, np.ndarray]:
        """Promedia la caída entre semillas y la pinta en cada cuadrado de la imagen."""
        promedio = tabla.groupby(["embedding", "fila", "columna"], sort=False)["caida"].mean()
        mapas = {}
        for (embedding, fila, columna), caida in promedio.items():
            mapa = mapas.setdefault(embedding, np.zeros((conjunto.alto, conjunto.ancho)))
            mapa[fila : fila + VENTANA_OCLUSION, columna : columna + VENTANA_OCLUSION] = caida
        return mapas

    @staticmethod
    def _guardar(tablas: dict[str, list[dict]]) -> None:
        for nombre, filas in tablas.items():
            ruta = DIR_RESULTADOS / f"extras_{nombre}.csv"
            pd.DataFrame(filas).round(6).to_csv(ruta, index=False)
            print(f"Tabla guardada en {ruta}")
