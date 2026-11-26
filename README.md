# Laboratorio 02: Comparación de Embeddings Faciales Lineales (PCA, LDA y Fisherfaces)

**Asignatura:** Machine Learning (2026-02) — Universidad Católica del Norte  
**Autores:** Martín Droguett Robledo y Amelie Valderrama Campos  
**Prototipo desplegado:** https://reconocimientofacial-md-av.streamlit.app/

Pipeline reproducible para evaluar y comparar técnicas lineales de extracción de características faciales (**PCA/Eigenfaces**, **LDA** y **Fisherfaces**) combinadas con tres clasificadores (**SVM lineal**, **$k$-NN** y **Árbol de Decisión**) sobre **Olivetti Faces** y **LFW (Top 50)** bajo un protocolo estricto sin fuga de datos (*data leakage*).

## 1. Requisitos e Instalación

### Entorno local (Conda — recomendado para experimentos)
```bash
conda env create -f environment.yml
conda activate lab-embeddings-faciales
```

Para actualizar el entorno si se modifican dependencias:
```bash
conda env update -f environment.yml --prune
```

*(Nota: El archivo `requirements.txt` se mantiene exclusivamente para el despliegue en servidor Linux de Streamlit Community Cloud).*

### Configuración opcional
Los parámetros por defecto (`random_state=42`, `test_size=0.25`, caché en `data/cache`) permiten reproducir todas las tablas del informe. Para personalizar variables locales sin exponer configuraciones privadas:
```bash
cp .env.example .env
```

## 2. Ejecución de Experimentos y Pruebas

Con el entorno `lab-embeddings-faciales` activado desde la raíz del repositorio:

```bash
# 1. Ejecutar suite de pruebas unitarias
pytest

# 2. Ejecutar protocolo experimental base (Olivetti y LFW Top 50)
python main.py experimentos

# 3. Ejecutar análisis de robustez entre semillas (42 a 46)
python main.py robustez

# 4. Ejecutar análisis extendido (iluminación, oclusión, desconocidos y barrido d_PCA)
python main.py extras

# 5. Ejecutar interfaz interactiva localmente
streamlit run app.py
```

También es posible ejecutar cada base de datos por separado mediante `python main.py olivetti` o `python main.py lfw`.

## 3. Salidas Generadas

* **`data/resultados/resultados_modelos.csv`**: Métricas (`accuracy`, `macro_f1`, tiempos de embedding y clasificación) de las 54 combinaciones evaluadas.
* **`data/resultados/predicciones_*.csv`**: Predicciones individuales sobre `TEST` para inspección de aciertos y errores.
* **`data/resultados/extras_*.csv`**: Resultados del análisis extendido (`iluminacion`, `oclusion`, `desconocidos`, `barrido_dpca`).
* **`figuras/`**: Curvas de varianza acumulada PCA, bases visuales (Eigenfaces y Fisherfaces), proyecciones 2D (t-SNE/UMAP), matrices de confusión y mapas de robustez.

## 4. Estructura del Proyecto

```text
main.py                 Orquestador CLI (olivetti, lfw, experimentos, robustez, extras)
app.py                  Aplicación interactiva en Streamlit
analisis.ipynb          Notebook de análisis exploratorio, tablas y auditoría de errores
environment.yml         Definición oficial del entorno Conda
requirements.txt        Dependencias para despliegue en Streamlit Community Cloud
src/
├── config.py           Constantes globales, semillas y rutas
├── pipeline.py         Orquestación de entrenamiento y evaluación sin fuga
├── extras.py           Experimentos extendidos de robustez y barrido d_PCA
├── datos/              Cargadores de Olivetti y LFW Top 50, y partición estratificada
├── embeddings/         Extractores PCA, LDA y Fisherfaces (PCA -> LDA)
├── clasificadores/     Fábrica de clasificadores (SVM lineal, k-NN, Árbol de Decisión)
├── evaluacion/         Métricas, perturbaciones y generación de gráficos
└── interfaz/           Servicio de inferencia para Streamlit
tests/                  Pruebas unitarias (pytest)
```

## 5. Consideraciones Éticas

Este proyecto utiliza exclusivamente conjuntos de datos públicos con fines académicos (Olivetti Faces y LFW). No se incorporan fotografías personales ni se infieren atributos demográficos. Los modelos lineales evaluados son sensibles a cambios de iluminación, pose y dominio, por lo que no son aptos para sistemas de vigilancia o control biométrico en entornos reales.