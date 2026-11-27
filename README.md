# Laboratorio 02: comparación de embeddings faciales lineales (PCA, LDA y Fisherfaces)

**Asignatura:** Machine Learning (2026-02), Universidad Católica del Norte  
**Autores:** Martín Droguett Robledo y Amelie Valderrama Campos  
**Prototipo desplegado:** https://reconocimientofacial-md-av.streamlit.app/

Pipeline reproducible para evaluar y comparar técnicas lineales de extracción de características faciales (**PCA/Eigenfaces**, **LDA** y **Fisherfaces**) combinadas con tres clasificadores (**SVM lineal**, **k-NN** y **árbol de decisión**) sobre **Olivetti Faces** y **LFW (Top 50)**, bajo un protocolo sin fuga de datos (*data leakage*): TRAIN aprende la media, las direcciones y el clasificador, y TEST solo pasa por `transform` y `predict`.

## 1. Requisitos e instalación

Se necesita Conda (Miniconda o Anaconda), Git y conexión a internet en la primera ejecución: el programa descarga Olivetti (cerca de 1 MB) y LFW (cerca de 300 MB en `data/cache/`).

```bash
git clone https://github.com/martindroguett/ML-Taller02.git
cd ML-Taller02
conda env create -f environment.yml
conda activate lab-embeddings-faciales
```

Para actualizar el entorno si se modifican dependencias:
```bash
conda env update -f environment.yml --prune
```

El proyecto no usa `pip install` ni `requirements.txt`. El despliegue en Streamlit Community Cloud también construye el entorno a partir de este `environment.yml`.

### Configuración opcional
Los parámetros por defecto (`random_state=42`, `test_size=0.25`, caché en `data/cache`) son los usados en el informe. Para cambiarlos sin modificar el código:
```bash
cp .env.example .env
```

## 2. Ejecución y verificación

Con el entorno `lab-embeddings-faciales` activado y desde la raíz del repositorio:

| Comando | Qué hace | Dónde verificar el resultado | Tiempo aprox. |
| --- | --- | --- | --- |
| `pytest` | 41 pruebas unitarias | `41 passed`. En un clon nuevo descarga Olivetti y omite la prueba de LFW hasta que LFW esté descargado (`40 passed, 1 skipped`) | segundos |
| `python main.py experimentos` | Protocolo completo en Olivetti y LFW: PCA, LDA y Fisherfaces con d = 10, 20, 30 y tres clasificadores | `data/resultados/resultados_modelos.csv` (54 filas), `predicciones_olivetti.csv` (2700) y `predicciones_lfw.csv` (18738); 72 figuras en `figuras/`; 54 modelos en `data/modelos/` | 3 a 4 min |
| `python main.py robustez` | Repite el protocolo con las semillas 42 a 46, sin figuras | `data/resultados/robustez_semillas.csv` (450 filas) y `robustez_resumen.csv` (90 filas) | 5 a 6 min |
| `python main.py extras` | Iluminación simulada, oclusión, identidades desconocidas y barrido de d_PCA, con las semillas 42 a 46 | `data/resultados/extras_*.csv` (4 archivos) y 8 figuras `figuras/extras_*.png` | 4 a 5 min |
| `streamlit run app.py` | Interfaz local sobre las 100 fotos de TEST de Olivetti | Se abre en `http://localhost:8501`. Si no encuentra los modelos en `data/modelos/`, los ajusta solo con TRAIN | — |

También se puede ejecutar cada base por separado con `python main.py olivetti` o `python main.py lfw`. Al cargar cada base, la consola muestra las formas de la partición:

- Olivetti: `X (400, 4096)  train (300, 4096)  test (100, 4096)  identidades 40  imagen 64x64`
- LFW: `X (2773, 2914)  train (2079, 2914)  test (694, 2914)  identidades 50  imagen 62x47`

Los tiempos se midieron en un portátil y varían según el equipo. Con la semilla 42 los resultados se repiten exactamente en un mismo equipo. Entre dos computadores distintos observamos diferencias de hasta 0.02 en el árbol de decisión; el resto de las métricas coincidió.

## 3. Salidas generadas

* **`data/resultados/resultados_modelos.csv`**: accuracy, macro-F1, tiempo de ajuste y predicción del clasificador (`tiempo_s`) y tiempo del embedding (`tiempo_embedding_s`) de las 54 combinaciones.
* **`data/resultados/predicciones_*.csv`**: identidad real y predicha de cada foto de TEST, para revisar aciertos y errores.
* **`data/resultados/robustez_*.csv`**: resultados por semilla y su media ± desviación.
* **`data/resultados/extras_*.csv`**: resultados del análisis extendido (`iluminacion`, `oclusion`, `desconocidos`, `barrido_dpca`).
* **`data/modelos/*.joblib`**: extractor y clasificador ajustados en TRAIN, que reutiliza la interfaz.
* **`figuras/`**: curvas de varianza acumulada de PCA, eigenfaces y fisherfaces, mapas t-SNE y UMAP de TRAIN, matrices de confusión y las figuras del análisis extendido.

Estas carpetas se regeneran con los comandos anteriores y no se suben a Git.

## 4. Notebook de análisis

`analisis.ipynb` reúne las características de los conjuntos, las tablas de resultados, las curvas de PCA, un mapa t-SNE, las fotos mal clasificadas, la robustez entre semillas y el análisis extendido. Lee los archivos de `data/resultados/` y `figuras/`, así que primero hay que ejecutar `experimentos`, `robustez` y `extras`. Se abre desde la raíz del repositorio con VS Code (extensión Jupyter) o Jupyter, usando el kernel del entorno `lab-embeddings-faciales`.

## 5. Estructura del proyecto

```text
main.py                 Orquestador CLI (olivetti, lfw, experimentos, robustez, extras)
app.py                  Aplicación interactiva en Streamlit
analisis.ipynb          Notebook de análisis de resultados y errores
environment.yml         Definición del entorno Conda
src/
├── config.py           Constantes globales, semillas y rutas
├── modelos.py          Contratos de datos (conjunto, partición y resultados)
├── pipeline.py         Protocolo sin fuga y estudio de robustez entre semillas
├── extras.py           Análisis extendido (iluminación, oclusión, desconocidos y barrido d_PCA)
├── datos/              Cargadores de Olivetti y LFW Top 50, y particiones
├── embeddings/         Extractores PCA, LDA y Fisherfaces (PCA -> LDA)
├── clasificadores/     Fábrica de clasificadores (SVM lineal, k-NN, árbol de decisión)
├── evaluacion/         Métricas, perturbaciones y gráficos
└── interfaz/           Servicio de inferencia para Streamlit
tests/                  Pruebas unitarias (pytest)
```

## 6. Aportes del grupo

La base del laboratorio corresponde al commit `4a25df2` del curso.

| Origen | Qué incluye | Archivos |
| --- | --- | --- |
| Provisto por el curso | Carga de Olivetti, partición estratificada, PCA, LDA, clasificadores, métricas, figuras e interfaz | `src/datos/olivetti.py`, `src/datos/division.py`, `src/embeddings/pca.py`, `src/embeddings/lda.py`, `src/clasificadores/`, `src/evaluacion/`, `src/interfaz/`, `app.py`, `main.py` |
| Implementado (pendiente del laboratorio) | Fisherfaces: PCA y LDA ajustados solo con TRAIN, con los controles d_LDA ≤ min(C−1, d_PCA) y d_PCA ≤ N−C. Cargador de LFW: 50 identidades con más fotos, desempate por orden alfabético y reindexado | `src/embeddings/fisherfaces.py`, `src/datos/lfw.py` |
| Modificado | Fisherfaces habilitado en el pipeline y en la interfaz; la interfaz decide la identidad con `predict`, igual que la tabla; predicciones por foto; tiempo del embedding; tiempos como mediana de 5 repeticiones; figura de fisherfaces; confusión normalizada en LFW; estudio de robustez; entorno con versiones fijas | `src/pipeline.py`, `src/interfaz/servicio.py`, `app.py`, `src/evaluacion/`, `src/config.py`, `main.py`, `environment.yml` |
| Nuevo | Análisis extendido, pruebas unitarias y notebook | `src/extras.py`, `src/evaluacion/perturbaciones.py`, `tests/`, `analisis.ipynb` |

## 7. Consideraciones éticas

Este proyecto utiliza exclusivamente conjuntos de datos públicos con fines académicos (Olivetti Faces y LFW). No se incorporan fotografías personales ni se infieren atributos demográficos. Los modelos lineales evaluados son sensibles a cambios de iluminación, pose y dominio, por lo que no son aptos para sistemas de vigilancia o control biométrico en entornos reales.
