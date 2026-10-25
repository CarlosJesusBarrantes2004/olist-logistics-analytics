import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
import xgboost as xgb

print("=== 1. PREPARACIÓN DE DATOS (CLASIFICACIÓN) ===")
df = pd.read_csv("../data/olist_master_dataset.csv")

# Recalculamos el retraso en días y creamos nuestra variable objetivo (Target) binaria
df["retraso_dias"] = df["dias_envio_real"] - df["dias_envio_estimado"]
# 1 = Riesgo (Retraso > 3 días), 0 = Normal
df["retraso_critico"] = np.where(df["retraso_dias"] > 3, 1, 0)

features = [
    "price",
    "freight_value",
    "product_weight_g",
    "customer_state",
    "product_category",
]
target_clf = "retraso_critico"

df_ml = df[features + [target_clf]].dropna()

top_cats = df_ml["product_category"].value_counts().head(10).index
df_ml["product_category"] = np.where(
    df_ml["product_category"].isin(top_cats), df_ml["product_category"], "Other"
)

X = pd.get_dummies(df_ml[features], drop_first=True)
y = df_ml[target_clf]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n=== 2. COMPARACIÓN DE MODELOS CLASIFICADORES ===")
modelos_clf = {
    "Logistic Regression": LogisticRegression(class_weight="balanced", max_iter=100),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    ),
    "XGBoost": xgb.XGBClassifier(
        n_estimators=150,
        learning_rate=0.1,
        max_depth=6,
        scale_pos_weight=5,
        random_state=42,
        n_jobs=-1,
    ),
}

resultados_clf = []
mejor_modelo = None
mejor_f1 = 0

for nombre, modelo in modelos_clf.items():
    print(f"Entrenando {nombre}...")
    modelo.fit(X_train_scaled, y_train)
    preds = modelo.predict(X_test_scaled)

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)

    resultados_clf.append(
        {"Modelo": nombre, "Accuracy": acc, "Precision": prec, "F1-Score": f1}
    )

    # Guardamos el mejor modelo basado en F1-Score para graficarlo luego
    if f1 > mejor_f1:
        mejor_f1 = f1
        mejor_modelo = (nombre, modelo)

print("\n=== 3. MÉTRICAS DE CLASIFICACIÓN ===")
df_res_clf = pd.DataFrame(resultados_clf).sort_values(by="F1-Score", ascending=False)
print(df_res_clf.to_string(index=False))

print(f"\n=== 4. MATRIZ DE CONFUSIÓN ({mejor_modelo[0]}) ===")
# Generamos la predicción final con el modelo ganador
pred_finales = mejor_modelo[1].predict(X_test_scaled)
matriz = confusion_matrix(y_test, pred_finales)

plt.figure(figsize=(6, 5))
sns.heatmap(
    matriz,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["A Tiempo (0)", "Retraso > 3 (1)"],
    yticklabels=["A Tiempo (0)", "Retraso > 3 (1)"],
)
plt.title(f"Matriz de Confusión - {mejor_modelo[0]}")
plt.xlabel("Predicción del Modelo")
plt.ylabel("Valor Real")
plt.tight_layout()
plt.show()
