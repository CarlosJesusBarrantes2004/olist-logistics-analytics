import pandas as pd
import numpy as np

print("=== 1. CARGANDO LOS DATASETS PRINCIPALES ===")
# Leer los 4 archivos clave (Ajusta la ruta 'data/' si los tienes en la misma carpeta)
df_orders = pd.read_csv("../data/olist_orders_dataset.csv")
df_items = pd.read_csv("../data/olist_order_items_dataset.csv")
df_customers = pd.read_csv("../data/olist_customers_dataset.csv")
df_products = pd.read_csv("../data/olist_products_dataset.csv")

print(f"Órdenes originales: {df_orders.shape[0]:,}")

print("\n=== 2. CRUZANDO TABLAS (EL PIPELINE DE MERGE) ===")
# Paso A: Unir Órdenes con los Ítems (Productos comprados y sus precios)
# Relación: 1 orden puede tener varios ítems. Llave: 'order_id'
df_master = pd.merge(df_orders, df_items, on="order_id", how="inner")

# Paso B: Agregar la información del Cliente (Estado/Provincia)
# Relación: 1 orden pertenece a 1 cliente. Llave: 'customer_id'
df_master = pd.merge(df_master, df_customers, on="customer_id", how="inner")

# Paso C: Agregar el nombre de la categoría del Producto
# Relación: 1 ítem es de 1 producto. Llave: 'product_id'
df_master = pd.merge(df_master, df_products, on="product_id", how="left")

print(f"Filas en la Tabla Maestra tras cruces: {df_master.shape[0]:,}")

print(f"\nInformación del dataset:\n")

df_master.info()

print(f"\nNúmero de nulos por columna:\n")

print(df_master.isnull().sum())

filas_con_nulos = df_master.isnull().any(axis=1).sum()

print(f"\nTotal de filas con algún nulo: {filas_con_nulos:,}")

print(f"\nFilas con nulos:\n")

df_con_nulos = df_master[df_master.isnull().any(axis=1)]

print(df_con_nulos.head())

print("\n=== 3. LIMPIEZA DE DATOS (DATA CLEANSING) ===")

print(df_master["order_status"].unique())

# Nos enfocamos solo en las órdenes que sí fueron entregadas
df_master = df_master[df_master["order_status"] == "delivered"]

# Las columnas de fechas vienen como texto (String). Hay que pasarlas a Datetime
columnas_fecha = [
    "order_purchase_timestamp",
    "order_delivered_customer_date",
    "order_estimated_delivery_date",
]

for col in columnas_fecha:
    df_master[col] = pd.to_datetime(df_master[col])

# Eliminar nulos críticos (si no hay fecha de entrega, no podemos medir logística)
df_master = df_master.dropna(subset=["order_delivered_customer_date"])

print("\n=== 4. CARGANDO NUEVOS ARCHIVOS ===")
df_reviews = pd.read_csv("../data/olist_order_reviews_dataset.csv")
df_translation = pd.read_csv("../data/product_category_name_translation.csv")

print("\n=== 5. INTEGRANDO SATISFACCIÓN DEL CLIENTE (REVIEWS) ===")
# Una misma orden puede tener varias reseñas si el cliente actualizó su queja.
# Para evitar duplicar filas en nuestra tabla maestra, agrupamos las reseñas
# por 'order_id' calculando el promedio de sus estrellas.
reviews_limpias = (
    df_reviews.groupby("order_id")
    .agg(review_score=("review_score", "mean"))
    .reset_index()
)

# Hacemos el cruce (LEFT JOIN) para no perder órdenes que no tengan reseña
df_master = pd.merge(df_master, reviews_limpias, on="order_id", how="left")

# Verificamos cuántas órdenes no tienen reseña (NaN)
sin_resena = df_master["review_score"].isna().sum()
print(f"Órdenes sin reseña que quedaron con valores nulos: {sin_resena:,}")

print("\n=== 6. TRADUCCIÓN DE CATEGORÍAS AL INGLÉS ===")

print("\nColumnas de 'df_translation':\n")

print(df_translation.columns)

# Cruzamos la tabla maestra con el diccionario de traducción
df_master = pd.merge(df_master, df_translation, on="product_category_name", how="left")

# Limpieza: Borramos la columna en portugués porque ya no la necesitamos
# y renombramos la columna en inglés para que quede más limpia.
if "product_category_name" in df_master.columns:
    df_master = df_master.drop(columns=["product_category_name"])

df_master = df_master.rename(
    columns={"product_category_name_english": "product_category"}
)

# Rellenamos los valores nulos en la categoría (productos sin categoría asignada)
df_master["product_category"] = df_master["product_category"].fillna("other/unknown")

print("\n=== 7. FEATURE ENGINEERING (CREANDO MÉTRICAS DE NEGOCIO) ===")
# Métrica 1: Días Reales que tardó el envío (Lo que experimentó el cliente)
df_master["dias_envio_real"] = (
    df_master["order_delivered_customer_date"] - df_master["order_purchase_timestamp"]
).dt.days

# Métrica 2: Días Estimados que prometió la tienda
df_master["dias_envio_estimado"] = (
    df_master["order_estimated_delivery_date"] - df_master["order_purchase_timestamp"]
).dt.days

# Métrica 3: ¿Llegó con retraso? (Retraso = Días Reales > Días Estimados)
# Usamos np.where (Si la diferencia es mayor a 0, hubo retraso)
df_master["retraso_dias"] = (
    df_master["dias_envio_real"] - df_master["dias_envio_estimado"]
)
df_master["llego_tarde"] = np.where(df_master["retraso_dias"] > 0, 1, 0)

print("\nVista previa de nuestras nuevas métricas logísticas:")
print(
    df_master[
        [
            "order_id",
            "product_category",
            "dias_envio_real",
            "dias_envio_estimado",
            "llego_tarde",
            "review_score",
        ]
    ].head()
)

print("\n=== 8. SEGMENTACIÓN RFM DE CLIENTES ===")
#  Definir la fecha actual (el día después de la última compra registrada en la base de datos)
fecha_referencia = df_master["order_purchase_timestamp"].max() + pd.Timedelta(days=1)

#  Calcular RFM agrupando por el ID ÚNICO del cliente
rfm = (
    df_master.groupby("customer_unique_id")
    .agg(
        # Recency: (Fecha de referencia - Fecha de su compra más reciente) en días
        Recency=("order_purchase_timestamp", lambda x: (fecha_referencia - x.max())),
        # Frequency: Cantidad de órdenes únicas (usamos nunique)
        Frecuency=("order_id", "nunique"),
        # Monetary: Suma de los precios de los productos comprados
        Monetary=("price", "sum"),
    )
    .reset_index()
)

#  Cruzar el score RFM de vuelta a nuestra Tabla Maestra (LEFT JOIN)
# Esto le dice a nuestro futuro modelo si el cliente es VIP o un comprador esporádico
df_master = pd.merge(df_master, rfm, on="customer_unique_id", how="left")

print("\n=== 9. GUARDANDO LA TABLA MAESTRA ===")
# Exportamos este dataframe limpio a un nuevo CSV para no tener que repetir
# todo este proceso de merge en las siguientes fases.
df_master.to_csv("../data/olist_master_dataset.csv", index=False)
print("¡Tabla Maestra guardada como 'olist_master_dataset.csv'!")
