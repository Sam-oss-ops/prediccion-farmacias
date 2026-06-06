import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 1. CONFIGURACIÓN DE LA INTERFAZ WEB
st.set_page_config(page_title="Predicción de Demanda - Farmacias", layout="wide")

st.title("📊 Sistema Inteligente de Predicción de Ventas Diarias")
st.markdown("---")

# 2. CARGA DEL MODELO PRE-ENTRENADO
import os
import gdown

@st.cache_resource
def cargar_modelo():
    archivo_destino = "modelo_random_forest.pkl"
    
    # Si el archivo no existe en el servidor de Streamlit, lo descarga una sola vez
    if not os.path.exists(archivo_destino):
        with st.spinner("Descargando el modelo predictivo desde el almacenamiento central (esto toma un momento)..."):
            # REEMPLAZA las XXXXXX de abajo con el ID real de tu archivo de Google Drive
            id_drive = "1-NGYPv50MFzd00ry3OcIp3XP9ujebQN9" 
            url_descarga = f"https://drive.google.com/uc?id={id_drive}"
            
            # Descargar de forma silenciosa el archivo de 1.5 GB
            gdown.download(url_descarga, archivo_destino, quiet=False)
            
    return joblib.load(archivo_destino)

try:
    modelo = cargar_modelo()
    st.sidebar.success("✅ Modelo Random Forest operativo")
except Exception as e:
    st.sidebar.error("❌ Error al inicializar el modelo")
    st.sidebar.write(f"Detalle técnico: {e}")



# 4. CARGA DEL NUEVO DATASET PARA PRUEBAS OPERATIVAS
st.header("2. Carga del Dataset de Evaluación")
archivo_cargado = st.file_uploader("Arrastra aquí tu archivo CSV o Excel con los datos de las sucursales para predecir:", type=["csv", "xlsx"])

if archivo_cargado is not None:
    if archivo_cargado.name.endswith('.csv'):
        df_prueba = pd.read_csv(archivo_cargado)
    else:
        df_prueba = pd.read_excel(archivo_cargado)
        
    st.subheader("📋 Vista previa de los datos ingresados")
    st.dataframe(df_prueba.head(5))
    
    # 5. PROCESAMIENTO DE PREDICCIONES
    st.header("3. Proyección Automatizada de Demanda")
    
    if st.button("🚀 Ejecutar Predicción de Ventas"):
        with st.spinner("El algoritmo está procesando los registros operativos..."):
            try:
                # El modelo realiza la estimación numérica en base a los datos cargados
                predicciones = modelo.predict(df_prueba)
                
                df_resultado = df_prueba.copy()
                df_resultado['Ventas_Predichas'] = np.round(predicciones, 2)
                
                st.success("¡Predicciones generadas exitosamente!")
                
                # Mostrar columnas esenciales junto al resultado final del dinero
                columnas_visibles = [col for col in ['Store', 'DayOfWeek', 'Promo', 'Ventas_Predichas'] if col in df_resultado.columns]
                st.dataframe(df_resultado[columnas_visibles].head(15))
                
                # Habilitar la descarga del archivo de salida
                csv_descarga = df_resultado.to_csv(index=False).encode('utf-8')
                st.download_button(label="📥 Descargar Resultados Completos en CSV", 
                                   data=csv_descarga, 
                                   file_name="predicciones_ventas_farmacia.csv", 
                                   mime="text/csv")
                    
            except Exception as e:
                st.error(f"Error en la alineación de características: {e}")
                st.info("Asegúrate de que las columnas del archivo coincidan exactamente con la matriz de entrenamiento.")
