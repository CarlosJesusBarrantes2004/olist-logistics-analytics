# 📦 E-Commerce Logistics & Delivery Prediction Engine

Este proyecto implementa un pipeline *end-to-end* de análisis de datos y Machine Learning sobre una base de datos relacional de e-commerce (+100k transacciones) extraída del [Brazilian E-Commerce Public Dataset de Olist en Kaggle](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce).

El sistema predice tiempos de entrega reales, evalúa riesgos de retraso y analiza el impacto operativo en la satisfacción del cliente.

---

## 📁 Fuente de Datos

- **Dataset:** [Brazilian E-Commerce Public Dataset by Olist (Kaggle)](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)
- **Descripción:** 100k órdenes entre 2016 y 2018 en Brasil, estructuradas en 9 tablas relacionales (clientes, pedidos, ítems, productos, pagos y reseñas).

---

## 🚀 Arquitectura del Proyecto

```text
olist-logistics-analytics/
├── src/                         # Datasets de Olist (excluidos por .gitignore)
    ├── etl_pipeline.py      # Pipeline ETL, consolidación de 9 tablas y RFM
    ├── exploratory_analysis.py    # EDA, tendencias temporales y degradation analysis
    ├── ml_regression.py          # Modelado regresor (MAE) y selección de mejor modelo
    ├── ml_classification.py      # Clasificación binaria de retrasos y matriz de confusión
├── app.py                        # Aplicación web interactiva en Streamlit
├── requirements.txt              # Dependencias del proyecto
└── README.md                     # Documentación del repositorio
```

## 📊 Hallazgos Clave de Analítica (EDA)

- **Impacto logístico:** Existe una caída del **45% en el nivel de satisfacción del cliente** (`review_score`) cuando un pedido sufre un retraso respecto a la fecha estimada (de ~4.3 a ~2.1 estrellas).
- **Cuellos de botella:** Categorías voluminosas (muebles, accesorios de oficina) representan los mayores tiempos promedio de despacho y flete.
- **Comportamiento RFM:** Integración de métricas de *Recency*, *Frequency* y *Monetary* agregadas a nivel de usuario único.

## 🤖 Modelado de Machine Learning

Se evaluaron y compararon múltiples algoritmos (`LinearRegression`, `RandomForest`, `XGBoost`) aplicando escalado dinámico (`StandardScaler`) y *One-Hot Encoding*.

### 1. Regresión (Predicción de Días Exactos)

- **Objetivo:** Predecir `dias_envio_real`.
- **Criterio de selección:** Menor **MAE (Error Absoluto Medio)**.
- **Resultado:** **XGBoost Regressor** logró un MAE de **~1.8 días**, superando a la Regresión Lineal y Random Forest.

### 2. Clasificación (Detección de Retrasos Críticos)

- **Objetivo:** Clasificar si un envío sufrirá un retraso > 3 días.
- **Métrica clave:** Balance de **Recall / F1-Score** aplicando ajuste de pesos por desbalance de clases (`scale_pos_weight`).

## 💻 Interfaz Web Interactiva (Streamlit)

El proyecto incluye un simulador en tiempo real para que los analistas de operaciones ingresen parámetros del producto (precio, flete, peso, estado y categoría) y obtengan estimaciones automáticas con el mejor modelo exportado (`.pkl`).

### Cómo ejecutar localmente

1. **Clonar el repositorio:**

   ```bash
   git clone https://github.com/TU_USUARIO/olist-logistics-analytics.git
   cd olist-logistics-analytics
   ```

2. **Instalar dependencias:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Ejecutar el pipeline y la app:**

   ```bash
   python etl_pipeline.py
   python ml_regression.py
   streamlit run app.py
   ```