"""SVM, k-NN y árbol con los mismos hiperparámetros en todo el laboratorio."""

from __future__ import annotations

import warnings

from sklearn.base import ClassifierMixin
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.config import CLASIFICADORES, RANDOM_STATE

warnings.filterwarnings(
    "ignore",
    message=r"The `probability` parameter was deprecated",
    category=FutureWarning,
)


class FabricaClasificadores:
    """Entrega un clasificador nuevo en cada llamada, listo para ``fit``."""

    def disponibles(self) -> tuple[str, ...]:
        return CLASIFICADORES

    def crear(self, nombre: str) -> ClassifierMixin:
        if nombre == "svm":
            return SVC(
                kernel="linear",
                C=1.0,
                probability=True,
                random_state=RANDOM_STATE,
            )
        if nombre == "knn":
            return KNeighborsClassifier(n_neighbors=3)
        if nombre == "arbol":
            return DecisionTreeClassifier(max_depth=8, random_state=RANDOM_STATE)
        raise ValueError(f"Clasificador desconocido: {nombre}")

    def parametros(self, nombre: str) -> str:
        texto = {
            "svm": "kernel=linear, C=1.0, probability=True",
            "knn": "n_neighbors=3, metric=euclidean",
            "arbol": "max_depth=8, criterion=gini",
        }
        if nombre not in texto:
            raise ValueError(f"Clasificador desconocido: {nombre}")
        return texto[nombre]

    def nota_probabilidad(self, nombre: str) -> str:
        """Qué significa el número que muestra la interfaz."""
        notas = {
            "svm": (
                "SVM lineal: la identidad se decide por el margen y la "
                "probabilidad se calibra aparte (escalado de Platt), así que "
                "la identidad predicha puede no ser la de mayor probabilidad."
            ),
            "knn": (
                "k-NN con k=3: la probabilidad es la fracción de vecinos "
                "más cercanos que votan por esa identidad."
            ),
            "arbol": (
                "Árbol con profundidad máxima 8: la probabilidad es la "
                "proporción de esa identidad en la hoja alcanzada."
            ),
        }
        if nombre not in notas:
            raise ValueError(f"Clasificador desconocido: {nombre}")
        return notas[nombre]
