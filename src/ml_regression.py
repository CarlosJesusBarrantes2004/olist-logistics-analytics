import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import joblib

print("=== 1. PREPARACIÓN DE DATOS (REGRESIÓN) ===")
df = pd.read_csv("../data/olist_master_dataset.csv")

features = [
    "price",
    "freight_value",
    "product_weight_g",
    "customer_state",
    "product_category",
]
target = "dias_envio_real"

df_ml = df[features + [target]].dropna()

# Reducción de dimensionalidad categórica
top_cats = df_ml["product_category"].value_counts().head(10).index
df_ml["product_category"] = np.where(
    df_ml["product_category"].isin(top_cats), df_ml["product_category"], "Other"
)

# Encoding y Split
X = pd.get_dummies(df_ml[features], drop_first=True)
y = df_ml[target]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Escalado
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n=== 2. ENTRENAMIENTO Y COMPARACIÓN DE MODELOS ===")
modelos = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(
        n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
    ),
    "XGBoost": xgb.XGBRegressor(
        n_estimators=150, learning_rate=0.1, max_depth=6, random_state=42, n_jobs=-1
    ),
}

resultados = []
mejor_mae = float("inf")
mejor_modelo = None

for nombre, modelo in modelos.items():
    print(f"Entrenando {nombre}...")
    modelo.fit(X_train_scaled, y_train)
    preds = modelo.predict(X_test_scaled)

    # Cálculo de métricas
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    resultados.append(
        {"Modelo": nombre, "MAE (Días)": mae, "RMSE (Días)": rmse, "R2 Score": r2}
    )

    # Lógica de Selección Dinámica: Si este modelo tiene menor error, se convierte en el ganador
    if mae < mejor_mae:
        mejor_mae = mae
        mejor_modelo = (nombre, modelo)

print("\n=== 3. TABLA COMPARATIVA FINAL ===")
df_resultados = pd.DataFrame(resultados).sort_values(by="MAE (Días)")
print(df_resultados.to_string(index=False))

print(
    f"\n🏆 Modelo Ganador Seleccionado: {mejor_modelo[0]} (MAE: {mejor_mae:.2f} días)"
)

print("\n=== 4. EXPORTANDO EL MEJOR MODELO PARA LA INTERFAZ WEB ===")
# Ahora exportamos dinámicamente el que haya ganado la competencia
joblib.dump(mejor_modelo[1], "mejor_modelo.pkl")
joblib.dump(scaler, "escalador.pkl")
joblib.dump(X_train.columns, "columnas_entrenamiento.pkl")
print(
    f"¡Archivos exportados exitosamente basados en el rendimiento de {mejor_modelo[0]}!"
)
