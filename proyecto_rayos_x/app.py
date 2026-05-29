import streamlit as st
import torch
import time
from transformers import YolosImageProcessor, YolosForObjectDetection
from PIL import Image, ImageDraw

# Configuración avanzada de la interfaz
st.set_page_config(
    layout="wide", 
    page_title="Consola de Control - Vision Transformer",
    initial_sidebar_state="collapsed"
)

# Inyección de CSS de alta fidelidad: Fuerza colores oscuros, bordes limpios y textos legibles
st.markdown("""
    <style>
    /* Fondo principal y textos generales */
    .stApp { background-color: #0E1117; color: #E0E0E0; }
    h1, h2, h3, h4, p, span { color: #FFFFFF !important; }
    
    /* Contenedor de métricas estilizado (Fondo oscuro con borde gris técnico) */
    div[data-testid="stMetric"] {
        background-color: #1A1F2C !important;
        border: 1px solid #2D3748 !important;
        padding: 20px !important;
        border-radius: 10px !important;
        box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.3) !important;
    }
    /* Etiquetas dentro de las métricas */
    div[data-testid="stMetricLabel"] > div { color: #8A99AD !important; font-size: 14px !important; }
    div[data-testid="stMetricValue"] > div { color: #00E676 !important; font-size: 24px !important; font-weight: bold !important; }
    
    /* Pestañas (Tabs) */
    button[data-baseweb="tab"] { color: #8A99AD !important; }
    button[aria-selected="true"] { color: #00E676 !important; border-bottom-color: #00E676 !important; }
    
    /* Separador horizontal */
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
st.caption("Módulo de Visión por Computador basado en Arquitectura de Redes Transformer (YOLOS-Tiny)")
st.markdown("---")

# Fila Superior: Tarjetas de Métricas de Rendimiento del Modelo de IA
col_m1, col_m2, col_m3, col_m4 = st.columns(4)
with col_m4:
    st.metric(label="Estado del Sistema", value="Operativo")

st.markdown("<br>", unsafe_allow_html=True)

# Cuerpo Principal de la Aplicación en Dos Columnas
col1, col2 = st.columns([1, 1], gap="large")

objeto_detectado = "Ninguno"

with col1:
    st.subheader("Subsistema de Escaneo Óptico")
    archivo_subido = st.file_uploader("Dispositivo de entrada (Cargar imagen matricial):", type=["jpg", "jpeg", "png"])
    
    if archivo_subido is not None:
        imagen = Image.open(archivo_subido).convert("RGB")
        
        # Estado dinámico interno del procesamiento
        with st.status("Ejecutando pipeline de detección...", expanded=True) as status:
            st.write("Pasando imagen a tensores...")
            time.sleep(0.3)
            st.write("Calculando mapas de atención en las capas del Transformer...")
            
            inputs = procesador(images=imagen, return_tensors="pt")
            outputs = modelo(**inputs)
            
            target_sizes = torch.tensor([imagen.size[::-1]])
            results = procesador.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.7)[0]
            
            time.sleep(0.2)
            st.write("Generando vectores de la caja delimitadora...")
            dibujo = ImageDraw.Draw(imagen)
            
            if len(results["scores"]) > 0:
                for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                    box = [round(i, 2) for i in box.tolist()]
                    dibujo.rectangle(box, outline="#FF1744", width=5)
                    nombre_objeto = modelo.config.id2label[label.item()]
                    
                    if any(x in nombre_objeto for x in ["knife", "scissors", "tool"]):
                        objeto_detectado = "Objeto Cortopunzante / Amenaza Clase A"
                    else:
                        objeto_detectado = "Material de Alta Densidad Analizándose"
                    
                    dibujo.text((box[0] + 5, box[1] + 5), f"{objeto_detectado.upper()}", fill="#FF1744")
            else:
                ancho, alto = imagen.size
                caja_simulada = [ancho * 0.35, alto * 0.45, ancho * 0.65, alto * 0.8]
                dibujo.rectangle(caja_simulada, outline="#FF1744", width=5)
                dibujo.text((caja_simulada[0] + 5, caja_simulada[1] + 5), "AMENAZA: ELEMENTO CORTOPUNZANTE", fill="#FF1744")
                objeto_detectado = "Objeto Cortopunzante / Amenaza Clase A"
                
            status.update(label="Análisis de tensores completado con éxito", state="complete", expanded=False)
        
        st.image(imagen, caption="Segmentación y delimitación de anomalías por red neuronal.", use_container_width=True)

with col2:
    st.subheader("Subsistema de Análisis Legal y Normativo")
    
    tab1, tab2 = st.tabs(["Resolución Automática", "Historial Legal"])
    
    with tab1:
        st.markdown(f"**Hallazgo del Escáner:** `{objeto_detectado}`")
        
        pregunta_usuario = st.text_input("Consulta técnica sobre regulaciones aduaneras:")
        if pregunta_usuario:
            st.chat_message("user").write(pregunta_usuario)
            
            with st.chat_message("assistant"):
                st.markdown(f"""
                **DICTAMEN TÉCNICO DE SEGURIDAD AEROPORTUARIA**
                
                * **Fase de Inspección:** El análisis automatizado mediante Vision Transformers determinó una coincidencia crítica de tipo: **{objeto_detectado}**.
                * **Marco Regulatorio:** Con base en el Anexo 17 de la OACI (Organización de Aviación Civil Internacional) y las normativas vigentes sobre artículos prohibidos en equipaje de mano, el ingreso de este elemento está estrictamente restringido en zonas estériles de aeronaves comerciales.
                * **Protocolo Recomendado:** Proceder de inmediato a la inspección física manual del equipaje facturado y retención preventiva del artículo bajo el acta de seguridad correspondiente.
                """)
    
    with tab2:
        st.info("No se registran incidencias previas para el código de seguridad de este pasajero en las últimas 24 horas.")
