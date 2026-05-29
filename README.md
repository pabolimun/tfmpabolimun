# TFM: Pablo Oliva Muñoz

> **Máster en Ingeniería del Software: Cloud, Datos y Gestión TI**  
> Universidad de Sevilla 
Curso: 2025/2026

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-Geometric-EE4C2C?logo=pytorch&logoColor=white)
![Colab](https://img.shields.io/badge/Google-Colab-F9AB00?logo=googlecolab&logoColor=white)
![TCGA](https://img.shields.io/badge/Data-TCGA--HNSCC-4CAF50)

---

## Descripción

Clasificación binaria de supervivencia (corta ≤450 días / larga >450 días) en pacientes fallecidos con carcinoma escamoso de cabeza y cuello (HNSCC) a partir de la cohorte TCGA-HNSCC.

El modelo opera sobre un **grafo heterogéneo** con 590 nodos miRNA y 5.305 isoformas de ARN mensajero, conectados mediante aristas de co-expresión Spearman y represión derivadas de TargetScan, integrando además el estado HPV como covariable clínica. La clasificación se realiza sobre los **198 pacientes fallecidos** de la cohorte con datos completos en ambas modalidades ómicas.

**AUC media: 0.648 · Mejor fold: 0.712**

---

## Estructura del repositorio

```text
├── Notebooks/
│   ├── NBFASE01_curacion_clinica_supervivencia.ipynb     # Curación clínica y construcción de la cohorte
│   ├── NBFASE02_preprocesamiento_multimodal.ipynb        # Preprocesamiento de miRNA e isoformas
│   ├── NBFASE03_construccion_supragrafo.ipynb            # Construcción del grafo heterogéneo
│   ├── NBFASE04_modelado_e_interpretabilidad.ipynb       # Modelado, entrenamiento e interpretabilidad
│   ├── NBFASE04_1_modelado_etiquetado.ipynb              # Comparativa de esquemas de etiquetado
│   └── NBFASE04_2_baseline_study.ipynb                   # Comparativa con arquitecturas de referencia
├── src/
│   └── hnsc_graph/
│       ├── gnn_models.py                                 # Arquitectura principal
│       ├── gnn_models_v2.py                              # Versión alternativa del modelo con distintas etiquetas de clasificación
│       ├── baseline_models.py                            # MLP, GCN y VanillaGAT
│       └── graph_construction.py                         # Construcción del grafo heterogéneo
└── data/
    └── (ver enlace a continuación)
```

---

## Datos

Los datos de entrada y los artefactos generados por el pipeline están disponibles en Google Drive debido a su tamaño.

[Acceder a los datos](https://drive.google.com/drive/folders/1bke4ZIqsnbiv_OBOc4qZYvZpEa6wem49?usp=sharing)

---

## Uso

El código está desarrollado para ejecutarse en **Google Colab**. Los notebooks siguen el orden del pipeline y deben ejecutarse secuencialmente. Es necesario subir los datos a Google Drive y ajustar las rutas de acceso si es necesario.
