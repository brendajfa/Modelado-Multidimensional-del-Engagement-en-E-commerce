# Modelado-Multidimensional-del-Engagement-en-E-commerce

Este repositorio contiene el código y la documentación del Trabajo de Fin de Máster (TFM) titulado: **"Modelado Multidimensional del Engagement en E-commerce: Un Enfoque de Clustering Espectral basado en Correlaciones Ordinales"**, presentado para el Máster en Ciencia de los Datos (Data Science) en **La Salle - Universidad Ramon Llull**. Propone un modelo multidimensional del engagement en e-commerce usando Clustering Espectral y correlaciones Kendall tau . Analiza 67 millones de eventos para identificar patrones en categorías, usuarios y tiempo , validando su estabilidad y utilidad para la toma de decisiones estratégicas


## 👤 Autora
*   **Alumna:** Brenda Juliana Fernández Alayo
*   **Ponente:** Jordi Escayola

## 📝 Resumen del Proyecto
El proyecto aborda la complejidad del comportamiento del usuario en una plataforma de e-commerce real mediante un enfoque de aprendizaje no supervisado. A diferencia de las métricas tradicionales, este modelo utiliza **Clustering Espectral** sobre una matriz de similitud basada en **correlaciones ordinales de Kendall tau**, permitiendo capturar relaciones no lineales en datos de comportamiento.

### Características principales:
*   **Dataset Masivo:** Análisis de 67 millones de eventos y 3.7 millones de usuarios únicos (Noviembre y Diciembre 2019).
*   **Modelado Multidimensional:** Se implementaron tres modelos complementarios:
    1.  **Modelo General:** Estructura global del catálogo y popularidad.
    2.  **Modelo User-based:** Perfiles basados en la lealtad y recurrencia (Tourist a VIP).
    3.  **Modelo Time-based:** Variaciones de comportamiento entre días laborables y fines de semana.
*   **Validación Robusta:** Evaluación estructural (Silhouette, Calinski-Harabasz) y validación funcional mediante la **proyección de datos de diciembre** sobre la geometría latente aprendida en noviembre, demostrando estabilidad temporal y resistencia al *concept drift* estacional.

## 🚀 Estructura del Código
El proyecto está dividido en notebooks que siguen el flujo de la metodología:
1.  **EDA:** Análisis exploratorio de 129 categorías y distribuciones de precio/eventos.
2.  **Preprocessing:** Agregación de métricas de popularidad, actividad y lealtad, transformación *rank-dense* y normalización Z-score.
3.  **Modeling:** Implementación del pipeline de Spectral Clustering con múltiples inicializaciones para asegurar robustez.
4.  **Business Impact:** Vinculación de los clusters con métricas financieras como el **GMV (Gross Merchandise Value)** y el análisis del embudo de conversión.

```
├── data/
│   ├── categories/         # Carpetas de categorías por conjunto
│   │   ├── evaluation_201912
│   │   └── training_201911
│   ├── processed/          # Bases agregadas y normalizadas
│   │   ├── training_201911
│   │   └── validation_201912
│   └── raw/                # Datos originales procesados / fuentes
│       ├── 2019-Dec-cleaned.parquet   # Vacío
│       ├── 2019-Nov-cleaned.parquet   # Vacío
│       └── source.txt
├── docs/               # Documentación oficial
│   └── Modelado Multidimensional del Engagement en E-commerce Un Enfoque de Clustering Espectral basado en Correlaciones Ordinales.pdf
├── models/               # Modelos, artefactos o checkpoints
├── modulos_comprimidos/  # Wheels generados
├── notebooks/
│   ├── 00_Load_Data.ipynb
│   ├── 01_EDA_Exploratory_Data_Analysis.ipynb
│   ├── 02_Aggregation_Models.ipynb
│   ├── 03_Model&Projection_General.ipynb
│   ├── 04_Model&Projection_User_Based.ipynb
│   ├── 05_Model&Projection_Time_Based.ipynb
│   └── 06_Validation.ipynb
├── results/
│   ├── clusters/
│   ├── EDA/
│   └── validation_analysis/
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_loader.py
│   ├── plots.py
│   ├── evaluation/
│   │   ├── absence_time.py
│   │   ├── gmv.py
│   │   └── kpis_analysis.py
│   ├── modelling/
│   │   ├── __init__.py
│   │   ├── similarity.py
│   │   ├── spectral.py
│   │   └── transforms.py
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── general.py
│   │   ├── time_based.py
│   │   └── user_based.py
│   └── utils/
│       ├── metrics.py
│       └── style.py
├── .gitignore          # Para evitar subir archivos temporales o datos pesados
├── requirements.txt    # Librerías necesarias (pandas, scikit-learn, etc.)
└── README.md           # Descripción principal del proyecto
```



## 📊 Resultados Clave
*   Identificación de **3 perfiles de engagement** en el modelo general y **5 perfiles** en el modelo de usuario.
*   Detección de categorías con "Engagement Críticamente Bajo" que actúan como puntos de fuga en el embudo.
*   Confirmación de que la geometría espectral aprendida es estable, mejorando los índices de validación incluso ante el cambio de comportamiento en la campaña navideña.

## 🛠️ Requisitos
Para replicar el entorno de análisis:
```bash
pip install -r requirements.txt