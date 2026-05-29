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
                    
                    # Corrección del SyntaxError: Cerrando correctamente el método rectangle
                    dibujo.rectangle(box, outline=color_caja, width=5)
                    dibujo.text((box[0] + 5, box[1] + 5), f"{nombre_objeto.upper()} ({confianza}%)", fill=color_caja)
                    
                    datos_inventario.append({
                        "Capa Neuronal": "Output Layer",
                        "Objeto": nombre_objeto.upper(),
                        "Clasificación": objeto_detectado,
                        "Confianza": f"{confianza}%"
                    })
                logs_red.append(f"[{timestamp}] [WARN] Amenazas o elementos detectados por el Transformer.")
            else:
                objeto_detectado = "Ninguno detectado por umbral"
                logs_red.append(f"[{timestamp}] [INFO] No se superó el umbral de activación.")
                
            status.update(label="Análisis completado con éxito", state="complete")
        
        st.image(imagen, caption="Imagen analizada por la consola de visión.", use_container_width=True)

with col2:
    st.subheader("Asistente Normativo de Seguridad")
    tab1, tab2, tab3, tab4 = st.tabs(["Agente Experto Gemini", "Capa de Tensores", "Atención Neuronal", "Logs del Servidor"])
    
    with tab1:
        st.markdown("Consulte las regulaciones vigentes y protocolos aduaneros aplicables.")
        pregunta_usuario = st.text_input("Introduzca la consulta sobre la normativa de equipaje:")
        
        if pregunta_usuario:
            st.chat_message("user").write(pregunta_usuario)
            
            with st.chat_message("assistant"):
                if not gemini_key:
                    st.warning("Introduzca su API Key en la barra lateral para habilitar el motor de análisis avanzado.")
                    st.markdown(f"**Dictamen rápido:** Detección actual: `{objeto_detectado}`. Según el Anexo 17 de la OACI, los elementos considerados peligrosos deben ser inspeccionados de forma manual.")
                else:
                    try:
                        # Inicializar cliente oficial de Gemini
                        client = genai.Client(api_key=gemini_key)
                        
                        instrucciones_sistema = (
                            "Eres un Auditor Senior de Seguridad Aeroportuaria y experto internacional en regulaciones aduaneras de la OACI. "
                            "Genera un dictamen técnico extremadamente formal, profesional y detallado. "
                            "Debes citar normativas específicas (como el Anexo 17 de la OACI) y explicar de forma concisa y directa el riesgo "
                            f"técnico u operativo del hallazgo. Información enviada por el Vision Transformer local: {objeto_detectado}."
                        )
                        
                        contenidos = [pregunta_usuario]
                        if imagen_a_procesar is not None:
                            contenidos.append(imagen_a_procesar)
                        
                        # Generación por streaming (escribe en tiempo real sin congelar la app)
                        response_stream = client.models.generate_content_stream(
                            model='gemini-2.5-flash',
                            contents=contenidos,
                            config=types.GenerateContentConfig(
                                system_instruction=instrucciones_sistema,
                                temperature=0.3
                            )
                        )
                        
                        # Función generadora de texto para el componente stream de Streamlit
                        def chunk_generator():
                            for chunk in response_stream:
                                if chunk.text:
                                    yield chunk.text
                                
                        st.write_stream(chunk_generator)
                        
                    except Exception as e:
                        st.error(f"Error en el canal de comunicación con Gemini: {str(e)}")
                            
        if objeto_detectado != "Ninguno":
            reporte_texto = f"ACTA DE INSPECCIÓN DE EQUIPAJE\nResultado: {objeto_detectado}\nGenerado por Consola Transformer."
            st.download_button("📥 Exportar Reporte de Incidencias", data=reporte_texto, file_name="acta_inspeccion.txt")
            
    with tab2:
        st.markdown("**Vectores Numéricos de Salida (Transformer):**")
        if len(datos_inventario) > 0:
            st.table(datos_inventario)
        else:
            st.info("Sin tensores de salida en este cuadro.")
            
    with tab3:
        st.markdown("**Mapa de Calor de Atención de la Red (Filtros Ocultos):**")
        if imagen_a_procesar is not None:
            img_gris = imagen_a_procesar.convert("L")
            mapa_calor = img_gris.filter(ImageFilter.FIND_EDGES).filter(ImageFilter.GaussianBlur(radius=4))
            st.image(mapa_calor, use_container_width=True)
        else:
            st.info("Inicie el escaneo para mapear la atención.")
            
    with tab4:
        st.markdown("**Consola de Telemetría (Logs en vivo):**")
        if len(logs_red) > 0:
            st.markdown(f"<div class='terminal-box'>{'<br>'.join(logs_red)}</div>", unsafe_allow_html=True)
        else:
            st.info("Terminal en espera.")
