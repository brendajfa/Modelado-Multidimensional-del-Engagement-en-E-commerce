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
├── data/               # (Nota: No subir los 67M de eventos, solo muestras o scripts de descarga)
│   ├── raw/            # Enlaces a los datasets originales (Kaggle)
│   └── processed/      # Bases agregadas (General, User-based, Time-based)
│       ├── 201911
│       └── 201912
├── docs/               # Documentación oficial
│   └── Modelado Multidimensional del Engagement en E-commerce Un Enfoque de Clustering Espectral basado en Correlaciones Ordinales.pdf
├── notebooks/          # Jupyter Notebooks organizados por etapas
│   ├── 01_EDA_Exploratory_Data_Analysis.ipynb
│   ├── 02_Preprocessing_and_Feature_Engineering.ipynb
│   ├── 03_Model_General_Spectral_Clustering.ipynb
│   ├── 04_Model_User_Based.ipynb
│   ├── 05_Model_Time_Based.ipynb
│   └── 06_Validation_and_Projection_Dec.ipynb
├── config.py               # Configuración (rutas de datos, hiperparámetros de K)
├── features/             # Definición de métricas (Popularidad, Actividad, Lealtad) [6]
├── src/                    # Carpeta con el núcleo lógico (el "motor" del TFM)
│   ├── __init__.py
│   ├── data_loader.py      # Carga y limpieza de los 67M de eventos [4]
│   ├── preprocessing.py    # Transformación Rank-dense y Z-score [7]
│   ├── clustering.py       # Implementación del Spectral Clustering y Kendall Tau [8, 9]
│   ├── evaluation       # Índices Silhouette, Calinski-Harabasz y Proyección [10, 11]
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── base.py           # Lógica común (Rank-dense y Z-score)
│   │   ├── general.py        # Agregación por categoría [3]
│   │   ├── user_based.py     # Lógica de segmentos (Tourist a VIP) [4]
│   │   └── time_based.py     # Ratios Weekday vs Weekend [5]
│   ├── features         # Ingeniería de métricas (Popularidad, Actividad, Lealtad) [5, 6]
│   ├── models/
│   │   └── spectral.py       # Pipeline: Kendall Tau -> Laplaciano -> Clustering [7, 8]
│   └── utils/
│       └── metrics.py        # Silhouette, Calinski-Harabasz [9, 10]
├── results/            # Gráficos y tablas generadas
│   ├── figures/        # Heatmaps, Dendrogramas, PCA plots
│   └── tables/         # Métricas de validación y GMV
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