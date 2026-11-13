"""Reconocimiento sobre fotografías que no entraron al entrenamiento."""

from __future__ import annotations

import joblib
import numpy as np

from src.clasificadores.fabrica import FabricaClasificadores
from src.config import DIMENSIONES, RANDOM_STATE, TEST_SIZE, asegurar_directorios, ruta_modelo
from src.datos.division import DivisorEstratificado
from src.datos.olivetti import CargadorOlivetti
from src.embeddings.fabrica import FabricaExtractores
from src.modelos import ConjuntoRostros, Particion, Prediccion


class ServicioReconocimiento:
    """Usa el mismo split y los mismos artefactos que ``main.py``."""

    def __init__(self) -> None:
        asegurar_directorios()
        self.conjunto: ConjuntoRostros = CargadorOlivetti().cargar()
        self.particion: Particion = DivisorEstratificado().dividir(self.conjunto)
        self.fabrica_extractores = FabricaExtractores()
        self.fabrica_clasificadores = FabricaClasificadores()
        self._memoria: dict[tuple, dict] = {}

    @property
    def dimensiones(self) -> tuple[int, ...]:
        return DIMENSIONES

    def etiquetas_prueba(self) -> list[str]:
        etiquetas = []
        for i, identidad in enumerate(self.particion.y_test):
            nombre = self._nombre(int(identidad))
            etiquetas.append(f"Prueba {i} · {nombre}")
        return etiquetas

    def imagen_prueba(self, indice: int) -> np.ndarray:
        self._exigir_indice(indice)
        return self.particion.images_test[indice]

    def predecir(self, indice: int, embedding: str, clasificador: str, d: int) -> Prediccion:
        self._exigir_indice(indice)
        bundle = self._obtener_bundle(embedding, clasificador, d)
        x = self.particion.X_test[indice : indice + 1]
        Z = bundle["extractor"].transform(x)
        modelo = bundle["clasificador"]
        probabilidad = np.asarray(modelo.predict_proba(Z)[0], dtype=float)
        clases = np.asarray(modelo.classes_)
        predicha = int(modelo.predict(Z)[0])
        ganador = int(np.flatnonzero(clases == predicha)[0])
        real = int(self.particion.y_test[indice])
        orden = np.argsort(probabilidad)[::-1][:5]
        top = [
            (self._nombre(int(clases[i])), float(probabilidad[i]))
            for i in orden
        ]
        return Prediccion(
            indice=indice,
            identidad_real=real,
            identidad_predicha=predicha,
            nombre_real=self._nombre(real),
            nombre_predicho=self._nombre(predicha),
            probabilidad=float(probabilidad[ganador]),
            acierto=predicha == real,
            top=top,
            nota_probabilidad=self.fabrica_clasificadores.nota_probabilidad(clasificador),
        )

    def _obtener_bundle(self, embedding: str, clasificador: str, d: int) -> dict:
        clave = (embedding, clasificador, d, RANDOM_STATE, TEST_SIZE)
        if clave in self._memoria:
            return self._memoria[clave]
        ruta = ruta_modelo(self.conjunto.nombre, embedding, clasificador, d)
        if ruta.exists():
            bundle = joblib.load(ruta)
            if (
                bundle.get("random_state") == RANDOM_STATE
                and bundle.get("test_size") == TEST_SIZE
            ):
                self._memoria[clave] = bundle
                return bundle
        bundle = self._entrenar(embedding, clasificador, d)
        self._memoria[clave] = bundle
        return bundle

    def _entrenar(self, embedding: str, clasificador: str, d: int) -> dict:
        extractor = self.fabrica_extractores.crear(embedding, d, self.conjunto.n_clases)
        extractor.fit(self.particion.X_train, self.particion.y_train)
        Z_train = extractor.transform(self.particion.X_train)
        modelo = self.fabrica_clasificadores.crear(clasificador)
        modelo.fit(Z_train, self.particion.y_train)
        bundle = {
            "extractor": extractor,
            "clasificador": modelo,
            "dataset": self.conjunto.nombre,
            "embedding": embedding,
            "modelo": clasificador,
            "d": extractor.dimension,
            "random_state": RANDOM_STATE,
            "test_size": TEST_SIZE,
            "nombres": self.conjunto.nombres_identidades,
            "alto": self.conjunto.alto,
            "ancho": self.conjunto.ancho,
        }
        joblib.dump(bundle, ruta_modelo(self.conjunto.nombre, embedding, clasificador, extractor.dimension))
        return bundle

    def _nombre(self, identidad: int) -> str:
        nombres = self.conjunto.nombres_identidades
        if 0 <= identidad < len(nombres):
            return nombres[identidad]
        return f"Identidad {identidad}"

    def _exigir_indice(self, indice: int) -> None:
        n = self.particion.images_test.shape[0]
        if indice < 0 or indice >= n:
            raise IndexError(f"La prueba {indice} no existe. Hay {n} fotografías de test.")
