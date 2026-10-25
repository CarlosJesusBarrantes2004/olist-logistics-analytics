import streamlit as st
import pandas as pd
import joblib

# 1. Configurar la página
st.set_page_config(page_title="Predicción de Envíos", page_icon="📦", layout="centered")
st.title("📦 Simulador de Tiempos de Entrega - Olist")
st.markdown(
    "Ingresa los datos de la compra para predecir cuántos días tardará en llegar al cliente."
)


# 2. Cargar los archivos que exportamos
@st.cache_resource
def cargar_modelos():
    modelo = joblib.load("src/mejor_modelo.pkl")
    scaler = joblib.load("src/escalador.pkl")
    columnas = joblib.load("src/columnas_entrenamiento.pkl")
    return modelo, scaler, columnas


modelo, scaler, columnas_entrenamiento = cargar_modelos()

# 3. Crear los controles visuales (Inputs)
col1, col2 = st.columns(2)

with col1:
    precio = st.number_input("Precio del Producto ($ USD)", min_value=1.0, value=50.0)
    flete = st.number_input("Costo del Flete ($ USD)", min_value=0.0, value=15.0)
    peso = st.number_input("Peso del Producto (Gramos)", min_value=100, value=1500)

with col2:
    estado = st.selectbox(
        "Estado del Cliente", ["SP", "RJ", "MG", "RS", "PR", "SC", "BA", "Other"]
    )
    categoria = st.selectbox(
        "Categoría del Producto",
        [
            "bed_bath_table",
            "health_beauty",
            "sports_leisure",
            "computers_accessories",
            "furniture_decor",
            "Other",
        ],
    )

# 4. Botón de Predicción
if st.button("🚀 Predecir Tiempo de Entrega"):
    # Convertimos lo que ingresó el usuario en un DataFrame de 1 fila
    entrada = pd.DataFrame(
        {
            "price": [precio],
            "freight_value": [flete],
            "product_weight_g": [peso],
            "customer_state": [estado],
            "product_category": [categoria],
        }
    )

    # Aplicamos el One-Hot Encoding igual que en el entrenamiento
    entrada_codificada = pd.get_dummies(entrada)

    # Alineamos las columnas para que coincidan exactamente con las que vio el modelo
    entrada_final = entrada_codificada.reindex(
        columns=columnas_entrenamiento, fill_value=0
    )

    # Escalamos los números
    entrada_escalada = scaler.transform(entrada_final)

    # Hacemos la predicción
    prediccion = modelo.predict(entrada_escalada)[0]

    # Mostramos el resultado visualmente
    st.success(f"🚚 Tiempo estimado de entrega: **{prediccion:.1f} días**")

    if prediccion > 15:
        st.warning(
            "⚠️ Atención: Este es un envío de alto riesgo por la distancia o el peso. Considere advertir al cliente."
        )
