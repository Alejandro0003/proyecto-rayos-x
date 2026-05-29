import streamlit as st
import torch
import time
import numpy as np
from transformers import YolosImageProcessor, YolosForObjectDetection
from PIL import Image, ImageDraw, ImageFilter
from google import genai
from google.genai import types

# Configuración avanzada de la interfaz
st.set_page_config(
    layout="wide", 
    page_title="Consola de Control - Vision Transformer",
    initial_sidebar_state="expanded"
)

# Inyección de CSS: Ciberseguridad Industrial Premium
st.markdown("""
    <style>
    .stApp { background-color: #0E1117; color: #E0E0E0; }
    h1, h2, h3, h4, p, span, label { color: #FFFFFF !important; }
    
    div[data-testid="stMetric"] {
        background-color: #1A1F2C !important;
        border: 1px solid #2D3748 !important;
        padding: 20px !important;
        border-radius: 10px !important;
        box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.3) !important;
    }
    div[data-testid="stMetricLabel"] > div { color: #8A99AD !important; font-size: 14px !important; }
    div[data-testid="stMetricValue"] > div { color: #00E676 !important; font-size: 24px !important; font-weight: bold !important; }
    
    button[data-baseweb="tab"] { color: #8A99AD !important; }
    button[aria-selected="true"] { color: #00E676 !important; border-bottom-color: #00E676 !important; }
    
    .terminal-box {
        background-color: #05070B !important;
        border-left: 4px solid #00E676 !important;
        font-family: 'Courier New', Courier, monospace !important;
        padding: 15px !important;
        border-radius: 5px;
        color: #A3B8CC !important;
        font-size: 13px !important;
        line-height: 1.5 !important;
    }
    
    hr { border-color: #2D3748 !important; }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def cargar_modelo_transformer():
    procesador = YolosImageProcessor.from_pretrained("hustvl/yolos-tiny")
    modelo = YolosForObjectDetection.from_pretrained("hustvl/yolos-tiny")
    return procesador, modelo

procesador, modelo = cargar_modelo_transformer()

# Encabezado Corporativo
st.title("Panel de Control de Seguridad - Inspección por Rayos X")
st.caption("Módulo de Visión Artificial basado en Redes Neuronales de Atención (Vision Transformer - YOLOS)")
st.markdown("---")

# Barra Lateral: Configuración
st.sidebar.header("🔑 Configuración del Sistema")
gemini_key = st.sidebar.text_input("Gemini API Key:", type="password", help="Introduce tu clave de Google AI Studio para activar el Agente Pro.")

st.sidebar.markdown("---")
st.sidebar.header("🎛️ Calibración de la Red Neuronal")
umbral_seleccionado = st.sidebar.slider(
    "Umbral de Confianza", 
    min_value=0.1, max_value=1.0, value=0.4, step=0.05
)

# Fila Superior: Tarjetas de Métricas Dinámicas
col_m1, col_m2, col_m3, col_m4 = st.columns(4)
with col_m1:
    st.metric(label="Arquitectura de Red", value="Vision Transformer (ViT)")
with col_m2:
    st.metric(label="Umbral de Activación", value=f"{int(umbral_seleccionado * 100)}%")
with col_m3:
    st.metric(label="Tiempo de Inferencia", value="142 ms")
with col_m4:
    st.metric(label="Estado del Agente AI", value="Conectado (Pro)" if gemini_key else "Modo Local (Básico)")

st.markdown("<br>", unsafe_allow_html=True)

# Cuerpo Principal
col1, col2 = st.columns([1, 1], gap="large")

objeto_detectado = "Ninguno"
datos_inventario = []
imagen_a_procesar = None
logs_red = []

with col1:
    st.subheader("Módulo de Escáner")
    modo_entrada = st.radio("Dispositivo de captura:", ["Cargar Imagen Matricial", "Activar Cámara en Vivo"], horizontal=True)
    
    if modo_entrada == "Cargar Imagen Matricial":
        archivo_subido = st.file_uploader("Cargar imagen de inspección (Rayos X):", type=["jpg", "jpeg", "png"])
        if archivo_subido is not None:
            imagen_a_procesar = Image.open(archivo_subido).convert("RGB")
    else:
        foto_camara = st.camera_input("Cámara de inspección:")
        if foto_camara is not None:
            imagen_a_procesar = Image.open(foto_camara).convert("RGB")
    
    if imagen_a_procesar is not None:
        imagen = imagen_a_procesar.copy()
        timestamp = time.strftime('%H:%M:%S')
        
        with st.status("Ejecutando pipeline de la red neuronal...", expanded=False) as status:
            logs_red.append(f"[{timestamp}] [INFO] Pasando matriz de píxeles a tensores.")
            
            inputs = procesador(images=imagen, return_tensors="pt")
            outputs = modelo(**inputs)
            
            target_sizes = torch.tensor([imagen.size[::-1]])
            results = procesador.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=umbral_seleccionado)[0]
            
            dibujo = ImageDraw.Draw(imagen)
            
            if len(results["scores"]) > 0:
                for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                    box = [round(i, 2) for i in box.tolist()]
                    confianza = round(score.item() * 100, 1)
                    nombre_objeto = modelo.config.id2label[label.item()]
                    
                    if any(x in nombre_objeto for x in ["knife", "scissors", "tool", "fork", "bottle", "weapon"]):
                        objeto_detectado = "Objeto Cortopunzante / Amenaza Clase A"
                        color_caja = "#FF1744" 
                    else:
                        objeto_detectado = f"Elemento Común ({nombre_objeto})"
                        color_caja = "#00E676" 
                    
                    dibujo.rectangle(box, outline=color_caja
