"""Orquestación OOP del laboratorio.

Olivetti y LFW siguen el mismo protocolo: PCA, LDA y Fisherfaces con tres
clasificadores, tabla de resultados, predicciones por foto y figuras.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.clasificadores.fabrica import FabricaClasificadores
from src.config import (
    DIMENSION_MAPAS,
    DIMENSIONES,
    DIR_RESULTADOS,
    RANDOM_STATE,
    SEMILLAS_ROBUSTEZ,
    TEST_SIZE,
    asegurar_directorios,
    ruta_modelo,
)
from src.datos.division import DivisorEstratificado
from src.datos.lfw import CargadorLFW
from src.datos.olivetti import CargadorOlivetti
from src.embeddings.base import ExtractorEmbedding
from src.embeddings.fabrica import FabricaExtractores
from src.embeddings.pca import ExtractorPCA
from src.evaluacion.metricas import Evaluador, cronometrar
from src.evaluacion.visualizacion import Visualizador
from src.modelos import ConjuntoRostros, Particion, ResultadoExperimento


class PipelineExperimentos:
    """Coordina la carga, el split, los embeddings y la evidencia en TEST."""

    def __init__(
        self,
        divisor: DivisorEstratificado | None = None,
        fabrica_extractores: FabricaExtractores | None = None,
        fabrica_clasificadores: FabricaClasificadores | None = None,
        evaluador: Evaluador | None = None,
        visualizador: Visualizador | None = None,
    ) -> None:
        self.divisor = divisor or DivisorEstratificado()
        self.fabrica_extractores = fabrica_extractores or FabricaExtractores()
        self.fabrica_clasificadores = fabrica_clasificadores or FabricaClasificadores()
        self.evaluador = evaluador or Evaluador()
        self.visualizador = visualizador or Visualizador()

    def ejecutar_olivetti(self) -> list[ResultadoExperimento]:
        print("== Experimento: Olivetti ==")
        return self._ejecutar(CargadorOlivetti().cargar())

    def ejecutar_lfw(self) -> list[ResultadoExperimento]:
        print("== Experimento: LFW ==")
        return self._ejecutar(CargadorLFW().cargar())

    def ejecutar_experimentos(self) -> None:
        self.ejecutar_olivetti()
        self.ejecutar_lfw()

    def ejecutar_robustez(self) -> None:
        """Repite el protocolo con varias semillas de split, sin figuras ni modelos.

        Además evalúa dos variantes de Fisherfaces con d_pca mayor que d_lda.
        """
        asegurar_directorios()
        filas: list[dict] = []
        for cargador in (CargadorOlivetti(), CargadorLFW()):
            conjunto = cargador.cargar()
            print(f"== Robustez: {conjunto.nombre} ==")
            for semilla in SEMILLAS_ROBUSTEZ:
                filas.extend(self._repetir(conjunto, semilla))
                print(f"  semilla {semilla} lista")
        self._guardar_robustez(pd.DataFrame(filas))

    def _ejecutar(self, conjunto: ConjuntoRostros) -> list[ResultadoExperimento]:
        asegurar_directorios()
        particion = self.divisor.dividir(conjunto)
        print(
            f"X {tuple(conjunto.X.shape)}  "
            f"train {tuple(particion.X_train.shape)}  "
            f"test {tuple(particion.X_test.shape)}  "
            f"identidades {conjunto.n_clases}  "
            f"imagen {conjunto.alto}x{conjunto.ancho}"
        )
        resultados: list[ResultadoExperimento] = []
        predicciones: list[pd.DataFrame] = []
        proyecciones_mapa: dict[tuple[str, int], tuple] = {}

        for extractor in self.fabrica_extractores.para_experimento(conjunto.n_clases):
            Z_train, Z_test, tiempo_embedding = self._proyectar(extractor, particion)
            if extractor.dimension == DIMENSION_MAPAS:
                proyecciones_mapa[(extractor.nombre, extractor.dimension)] = (
                    Z_train,
                    particion.y_train,
                )
            filas, tablas = self._evaluar_clasificadores(
                conjunto, extractor, particion, Z_train, Z_test, tiempo_embedding
            )
            resultados.extend(filas)
            predicciones.extend(tablas)

        self._figuras_pca(conjunto, particion)
        self._figura_fisherfaces(conjunto, particion)
        self._mapas(conjunto.nombre, proyecciones_mapa)
        self._guardar_csv(resultados, conjunto.nombre)
        self._guardar_predicciones(predicciones, conjunto.nombre)
        self._imprimir(resultados)
        return resultados

    def _proyectar(
        self,
        extractor: ExtractorEmbedding,
        particion: Particion,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Ajusta con TRAIN, proyecta TRAIN y TEST y mide cuánto tarda todo eso."""

        def ajustar_y_proyectar() -> tuple[np.ndarray, np.ndarray]:
            extractor.fit(particion.X_train, particion.y_train)
            return extractor.transform(particion.X_train), extractor.transform(particion.X_test)

        (Z_train, Z_test), tiempo_s = cronometrar(ajustar_y_proyectar, self.evaluador.repeticiones)
        return Z_train, Z_test, tiempo_s

    def _evaluar_clasificadores(
        self,
        conjunto: ConjuntoRostros,
        extractor: ExtractorEmbedding,
        particion: Particion,
        Z_train,
        Z_test,
        tiempo_embedding: float,
    ) -> tuple[list[ResultadoExperimento], list[pd.DataFrame]]:
        filas: list[ResultadoExperimento] = []
        tablas: list[pd.DataFrame] = []
        for nombre in self.fabrica_clasificadores.disponibles():
            modelo = self.fabrica_clasificadores.crear(nombre)
            medicion = self.evaluador.evaluar(
                modelo,
                Z_train,
                particion.y_train,
                Z_test,
                particion.y_test,
            )
            fila = ResultadoExperimento(
                dataset=conjunto.nombre,
                embedding=extractor.nombre,
                d=extractor.dimension,
                modelo=nombre,
                parametros=self.fabrica_clasificadores.parametros(nombre),
                accuracy=medicion.accuracy,
                macro_f1=medicion.macro_f1,
                tiempo_s=medicion.tiempo_s,
                tiempo_embedding_s=tiempo_embedding,
            )
            filas.append(fila)
            tablas.append(
                self._tabla_predicciones(
                    conjunto, extractor, particion, nombre, medicion.predicciones
                )
            )
            self._guardar_modelo(conjunto, extractor, nombre, modelo)
            ruta = self.visualizador.confusion(
                medicion.matriz,
                conjunto.nombre,
                extractor.nombre,
                nombre,
                extractor.dimension,
                normalizar=conjunto.desbalanceado,
            )
            print(
                f"  {extractor.nombre:12} d={extractor.dimension:<3} {nombre:6} "
                f"acc={fila.accuracy:.3f}  macro-F1={fila.macro_f1:.3f}  "
                f"{fila.tiempo_s:.2f}s  -> {ruta.name}"
            )
        return filas, tablas

    @staticmethod
    def _tabla_predicciones(
        conjunto: ConjuntoRostros,
        extractor: ExtractorEmbedding,
        particion: Particion,
        modelo: str,
        predicciones: np.ndarray,
    ) -> pd.DataFrame:
        """Una fila por fotografía de TEST: identidad real, predicha y si acertó."""
        nombres = np.asarray(conjunto.nombres_identidades)
        return pd.DataFrame(
            {
                "dataset": conjunto.nombre,
                "embedding": extractor.nombre,
                "d": extractor.dimension,
                "modelo": modelo,
                "indice_test": np.arange(particion.y_test.size),
                "indice_original": particion.indices_test,
                "y_real": particion.y_test,
                "y_pred": predicciones,
                "nombre_real": nombres[particion.y_test],
                "nombre_pred": nombres[predicciones],
                "acierto": predicciones == particion.y_test,
            }
        )

    def _guardar_modelo(self, conjunto, extractor, nombre: str, modelo) -> Path:
        ruta = ruta_modelo(conjunto.nombre, extractor.nombre, nombre, extractor.dimension)
        joblib.dump(
            {
                "extractor": extractor,
                "clasificador": modelo,
                "dataset": conjunto.nombre,
                "embedding": extractor.nombre,
                "modelo": nombre,
                "d": extractor.dimension,
                "random_state": RANDOM_STATE,
                "test_size": TEST_SIZE,
                "nombres": conjunto.nombres_identidades,
                "alto": conjunto.alto,
                "ancho": conjunto.ancho,
            },
            ruta,
        )
        return ruta

    def _figuras_pca(self, conjunto: ConjuntoRostros, particion: Particion) -> None:
        if conjunto.n_clases < 2:
            return
        d_ref = min(max(DIMENSIONES), particion.X_train.shape[0] - 1, particion.X_train.shape[1])
        referencia = ExtractorPCA(n_components=d_ref)
        referencia.fit(particion.X_train)
        self.visualizador.curva_pca(particion.X_train, conjunto.nombre)
        self.visualizador.eigenfaces(
            referencia.rostro_medio(conjunto.alto, conjunto.ancho),
            referencia.componentes_imagen(conjunto.alto, conjunto.ancho),
            conjunto.nombre,
        )

    def _figura_fisherfaces(self, conjunto: ConjuntoRostros, particion: Particion) -> None:
        referencia = self.fabrica_extractores.crear(
            "fisherfaces", max(DIMENSIONES), conjunto.n_clases
        )
        referencia.fit(particion.X_train, particion.y_train)
        self.visualizador.fisherfaces(
            referencia.componentes_imagen(conjunto.alto, conjunto.ancho),
            conjunto.nombre,
            referencia.d_pca,
            referencia.d_lda,
        )

    def _mapas(self, dataset: str, proyecciones: dict) -> None:
        for (embedding, d), (Z_train, y_train) in proyecciones.items():
            for metodo in ("tsne", "umap"):
                ruta = self.visualizador.mapa(Z_train, y_train, metodo, dataset, embedding, d)
                print(f"  mapa {ruta.name}")

    def _guardar_csv(self, filas: list[ResultadoExperimento], dataset: str) -> Path:
        columnas = [
            "dataset",
            "embedding",
            "d",
            "modelo",
            "parametros",
            "accuracy",
            "macro_f1",
            "tiempo_s",
            "tiempo_embedding_s",
        ]
        nuevo = pd.DataFrame([fila.__dict__ for fila in filas], columns=columnas)
        if not nuevo.empty:
            nuevo["accuracy"] = nuevo["accuracy"].round(6)
            nuevo["macro_f1"] = nuevo["macro_f1"].round(6)
            nuevo["tiempo_s"] = nuevo["tiempo_s"].round(4)
            nuevo["tiempo_embedding_s"] = nuevo["tiempo_embedding_s"].round(4)
        ruta = DIR_RESULTADOS / "resultados_modelos.csv"
        if ruta.exists():
            previo = pd.read_csv(ruta)
            previo = previo[previo["dataset"] != dataset]
            nuevo = pd.concat([previo, nuevo], ignore_index=True)
        nuevo.to_csv(ruta, index=False)
        print(f"Tabla guardada en {ruta}")
        return ruta

    def _guardar_predicciones(self, tablas: list[pd.DataFrame], dataset: str) -> Path:
        ruta = DIR_RESULTADOS / f"predicciones_{dataset}.csv"
        pd.concat(tablas, ignore_index=True).to_csv(ruta, index=False)
        print(f"Predicciones por foto en {ruta}")
        return ruta

    def _repetir(self, conjunto: ConjuntoRostros, semilla: int) -> list[dict]:
        """Una pasada del protocolo con otro split estratificado."""
        divisor = DivisorEstratificado(self.divisor.test_size, semilla)
        particion = divisor.dividir(conjunto)
        extractores = self.fabrica_extractores.para_experimento(conjunto.n_clases)
        extractores += self.fabrica_extractores.variantes_fisherfaces(
            conjunto.n_clases, particion.X_train.shape[0]
        )
        evaluador = Evaluador(repeticiones=1)
        filas: list[dict] = []
        for extractor in extractores:
            extractor.fit(particion.X_train, particion.y_train)
            Z_train = extractor.transform(particion.X_train)
            Z_test = extractor.transform(particion.X_test)
            for nombre in self.fabrica_clasificadores.disponibles():
                medicion = evaluador.evaluar(
                    self.fabrica_clasificadores.crear(nombre),
                    Z_train,
                    particion.y_train,
                    Z_test,
                    particion.y_test,
                )
                filas.append(
                    {
                        "dataset": conjunto.nombre,
                        "semilla": semilla,
                        "embedding": extractor.nombre,
                        "d": extractor.dimension,
                        "modelo": nombre,
                        "accuracy": medicion.accuracy,
                        "macro_f1": medicion.macro_f1,
                    }
                )
        return filas

    def _guardar_robustez(self, detalle: pd.DataFrame) -> None:
        """Guarda una fila por semilla y, aparte, media y desviación por combinación."""
        detalle = detalle.round({"accuracy": 6, "macro_f1": 6})
        detalle.to_csv(DIR_RESULTADOS / "robustez_semillas.csv", index=False)
        resumen = (
            detalle.groupby(["dataset", "embedding", "d", "modelo"], sort=False)
            .agg(
                semillas=("semilla", "nunique"),
                accuracy_media=("accuracy", "mean"),
                accuracy_std=("accuracy", "std"),
                macro_f1_media=("macro_f1", "mean"),
                macro_f1_std=("macro_f1", "std"),
            )
            .round(4)
            .reset_index()
        )
        ruta = DIR_RESULTADOS / "robustez_resumen.csv"
        resumen.to_csv(ruta, index=False)
        print(f"Resumen guardado en {ruta}")
        with pd.option_context("display.width", 120):
            print(resumen.to_string(index=False))

    @staticmethod
    def _imprimir(filas: list[ResultadoExperimento]) -> None:
        if not filas:
            print("Sin filas nuevas en la tabla.")
            return
        tabla = pd.DataFrame([fila.__dict__ for fila in filas])
        with pd.option_context("display.max_colwidth", 48, "display.width", 120):
            print(tabla.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
