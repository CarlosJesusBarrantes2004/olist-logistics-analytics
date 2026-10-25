import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.ticker as ticker

print("=== CARGANDO TABLA MAESTRA ===")
df = pd.read_csv("../data/olist_master_dataset.csv")

df.info()

# Configurar el estilo visual corporativo de Seaborn
sns.set_theme(style="whitegrid")

print("Generando Gráfico 1: Tasa de Retrasos...")
# -------------------------------------------------------------------------
# GRÁFICO 1: PROPORCIÓN DE RETRASOS (PIE CHART CON PANDAS)
# -------------------------------------------------------------------------
plt.figure(figsize=(6, 6))

# Contar cuántos llegaron tarde (1) vs a tiempo (0)
conteo_retrasos = df["llego_tarde"].value_counts()
etiquetas = ["A Tiempo", "Con Retraso"]
colores = ["#4CAF50", "#F44336"]

# Usamos el .plot.pie() directo de Pandas
conteo_retrasos.plot.pie(
    labels=etiquetas, colors=colores, autopct="%1.1f%%", startangle=90, ylabel=""
)
plt.title("Tasa Global de Retrasos en Envíos", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()

print("Generando Gráfico 2: Impacto en Satisfacción...")
# -------------------------------------------------------------------------
# GRÁFICO 2: IMPACTO DEL RETRASO EN LAS RESEÑAS (BARPLOT CON SEABORN)
# -------------------------------------------------------------------------
plt.figure(figsize=(8, 5))

# sns.barplot calcula automáticamente el PROMEDIO de 'review_score' para cada grupo
sns.barplot(data=df, x="llego_tarde", y="review_score", palette=["#4CAF50", "#F44336"])

# Personalización de las etiquetas
plt.xticks(ticks=[0, 1], labels=["A Tiempo", "Con Retraso"], fontsize=12)
plt.title(
    "Impacto Logístico en la Satisfacción del Cliente", fontsize=14, fontweight="bold"
)
plt.xlabel("Estado de la Entrega", fontsize=12)
plt.ylabel("Calificación Promedio (Estrellas)", fontsize=12)

# Establecemos el límite del eje Y de 1 a 5 (las estrellas posibles)
plt.ylim(1, 5)
plt.tight_layout()
plt.show()

print("Generando Gráfico 3: Cuellos de Botella por Categoría...")
# -------------------------------------------------------------------------
# GRÁFICO 3: TOP 10 CATEGORÍAS QUE MÁS TARDAN EN LLEGAR
# -------------------------------------------------------------------------
plt.figure(figsize=(10, 6))

# 1. Agrupamos por categoría y calculamos el promedio de días reales
retraso_categoria = (
    df.groupby("product_category")["dias_envio_real"].mean().reset_index()
)

# 2. Ordenamos de mayor a menor y tomamos las peores 10
top_peores = retraso_categoria.sort_values(by="dias_envio_real", ascending=False).head(
    10
)

# 3. Graficamos un gráfico de barras horizontal (sns.barplot)
sns.barplot(data=top_peores, x="dias_envio_real", y="product_category", color="#FF9800")

plt.title(
    "Top 10 Categorías con Mayor Tiempo de Entrega Promedio",
    fontsize=14,
    fontweight="bold",
)
plt.xlabel("Días Promedio de Entrega", fontsize=12)
plt.ylabel("Categoría de Producto", fontsize=12)
plt.tight_layout()
plt.show()

print("Generando Gráfico 4: Evolución Temporal de Ventas vs Retrasos...")
# -------------------------------------------------------------------------
# GRÁFICO 4: TENDENCIA MENSUAL (Ventas vs Tasa de Retraso) - Doble Eje Y
# -------------------------------------------------------------------------
# 1. Convertimos la fecha a formato Año-Mes (ej. '2017-05')
df["order_purchase_timestamp"] = pd.to_datetime(df["order_purchase_timestamp"])

df["mes_anio"] = df["order_purchase_timestamp"].dt.to_period("M").astype(str)

# 2. Agrupamos por mes para sacar Total de Ventas y % de Retrasos
tendencia = (
    df.groupby("mes_anio")
    .agg(Ventas_Totales=("price", "sum"), Tasa_Retraso=("llego_tarde", "mean"))
    .reset_index()
)

# Filtramos los primeros y últimos meses si están incompletos para no sesgar el gráfico
tendencia = tendencia[
    (tendencia["mes_anio"] >= "2017-01") & (tendencia["mes_anio"] <= "2018-08")
]

fig, ax1 = plt.subplots(figsize=(12, 6))

# Línea 1 (Eje Y izquierdo): Ventas Totales
color_ventas = "tab:blue"
ax1.set_xlabel("Mes del Año", fontsize=12)
ax1.set_ylabel("Ventas Totales ($)", color=color_ventas, fontsize=12)
ax1.plot(
    tendencia["mes_anio"],
    tendencia["Ventas_Totales"],
    color=color_ventas,
    marker="o",
    linewidth=2,
)
ax1.tick_params(axis="y", labelcolor=color_ventas)
plt.xticks(rotation=45)

ax1.get_yaxis().set_major_formatter(
    ticker.FuncFormatter(lambda x, p: format(int(x), ","))
)

# Línea 2 (Eje Y derecho): Tasa de Retrasos
ax2 = ax1.twinx()
color_retraso = "tab:red"
ax2.set_ylabel("Tasa de Retrasos (%)", color=color_retraso, fontsize=12)
ax2.plot(
    tendencia["mes_anio"],
    tendencia["Tasa_Retraso"] * 100,
    color=color_retraso,
    marker="x",
    linewidth=2,
    linestyle="dashed",
)
ax2.tick_params(axis="y", labelcolor=color_retraso)

plt.title(
    "Tendencia: ¿Los picos de ventas provocan más retrasos logísticos?",
    fontsize=14,
    fontweight="bold",
)
fig.tight_layout()
plt.show()

print("Generando Gráfico 5: Distribución de Fletes por Estado...")
# -------------------------------------------------------------------------
# GRÁFICO 5: BOXPLOTS DE COSTO FLETE POR ESTADO (TOP 5 ESTADOS)
# -------------------------------------------------------------------------
# 1. Identificamos los 5 estados con más ventas para que el gráfico sea legible
top_5_estados = df["customer_state"].value_counts().head(5).index
df_top_estados = df[df["customer_state"].isin(top_5_estados)]

plt.figure(figsize=(10, 6))

# 2. Generamos el Boxplot
sns.boxplot(data=df_top_estados, x="customer_state", y="freight_value", palette="Set2")

# Recortamos el eje Y en $100 para evitar que 2 o 3 fletes extremadamente caros (outliers) aplasten la visualización
plt.ylim(0, 100)
plt.title(
    "Distribución del Costo de Flete en los 5 Estados Principales",
    fontsize=14,
    fontweight="bold",
)
plt.xlabel("Estado del Cliente", fontsize=12)
plt.ylabel("Costo de Envío (Flete)", fontsize=12)
plt.tight_layout()
plt.show()

print("Generando Gráfico 6: Correlación Días de Retraso vs Satisfacción...")


# -------------------------------------------------------------------------
# GRÁFICO 6: IMPACTO DE LOS DÍAS DE RETRASO EN LA CALIFICACIÓN
# -------------------------------------------------------------------------
# 1. Clasificamos numéricamente los días de retraso en categorías (Bins)
def categoria_retraso(dias):
    if dias <= 0:
        return "A. A tiempo o antes"
    elif dias <= 3:
        return "B. Retraso Leve (1-3 días)"
    elif dias <= 7:
        return "C. Retraso Medio (4-7 días)"
    else:
        return "D. Retraso Crítico (+7 días)"


df["rango_retraso"] = df["retraso_dias"].apply(categoria_retraso)

plt.figure(figsize=(9, 5))

# 2. Seaborn calcula el promedio de estrellas por cada rango de retraso
sns.barplot(
    data=df.sort_values("rango_retraso"),
    x="rango_retraso",
    y="review_score",
    palette="magma",
)

plt.title(
    "Degradación de la Satisfacción según Gravedad del Retraso",
    fontsize=14,
    fontweight="bold",
)
plt.xlabel("Gravedad del Retraso", fontsize=12)
plt.ylabel("Calificación Promedio (1 a 5)", fontsize=12)
plt.ylim(1, 5)
plt.tight_layout()
plt.show()
