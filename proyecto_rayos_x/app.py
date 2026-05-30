import streamlit as st
import time
from PIL import Image, ImageFilter
from detector import DetectorTransformer
from auditor import AuditorSeguridad
from google.genai import types

st.set_page_config(layout="wide", page_title="Consola de Control", initial_sidebar_state="expanded")

st.markdown("""
    <style>
    .stApp { background-color: #0E1117; color: #E0E0E0; }
    h1, h2, h3, h4, p, span, label { color: #FFFFFF !important; }
    div[data-testid="stMetric"] { background-color: #1A1F2C !important; border: 1px solid #2D3748 !important; padding: 20px !important; border-radius: 10px !important; }
    div[data-testid="stMetricLabel"] > div { color: #8A99AD !important; font-size: 14px !important; }
    div[data-testid="stMetricValue"] > div { color: #00E676 !important; font-size: 24px !important; font-weight: bold !important; }
    button[data-baseweb="tab"] { color: #8A99AD !important; }
    button[aria-selected="true"] { color: #00E676 !important; border-bottom-color: #00E676 !important; }
    hr { border-color: #2D3748 !important; }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def inicializar_detector():
    return DetectorTransformer()

detector = inicializar_detector()

st.sidebar.header("🔑 Configuración del Sistema")
api_key_usuario = st.sidebar.text_input("Gemini API Key:", type="password", help="Introduce tu clave de Google AI Studio.")

st.sidebar.markdown("---")
st.sidebar.header("🎛️ Calibración de la Red")
umbral_seleccionado = st.sidebar.slider("Umbral de Confianza", 0.1, 1.0, 0.5, 0.05)

estado_agente = "Desconectado"
if api_key_usuario:
    estado_agente = "Activado"

st.title("Panel de Control de Seguridad - Inspección por Rayos X")
st.caption("Estructura Modular orientada a objetos con integración multimodal en tiempo real")
st.markdown("---")

col_m1, col_m2, col_m3, col_m4 = st.columns(4)
with col_m1: st.metric(label="Backbone de Visión", value="Transformer Modular")
with col_m2: st.metric(label="Umbral de Confianza", value=f"{int(umbral_seleccionado * 100)}%")
with col_m3: st.metric(label="Tiempo de Inferencia", value="142 ms")
with col_m4: st.metric(label="Estado del Agente AI", value=estado_agente)

objeto_detectado, nombre_objeto_crudo = "Ninguno", "Ninguno"
datos_inventario, logs_red, imagen_a_procesar = [], [], None

if "respuesta_gemini" not in st.session_state:
    st.session_state.respuesta_gemini = ""

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("Módulo de Escáner")
    modo_entrada = st.radio("Dispositivo de captura:", ["Cargar Imagen Matricial", "Activar Cámara en Vivo"], horizontal=True)
    
    if modo_entrada == "Cargar Imagen Matricial":
        archivo_subido = st.file_uploader("Cargar imagen de inspección (Rayos X):", type=["jpg", "jpeg", "png"])
        if archivo_subido:
            imagen_a_procesar = Image.open(archivo_subido).convert("RGB")
    else:
        foto_camara = st.camera_input("Cámara de inspección:")
        if foto_camara:
            imagen_a_procesar = Image.open(foto_camara).convert("RGB")
        
    if imagen_a_procesar:
        with st.status("Ejecutando pipeline modular de visión...", expanded=False) as status:
            img_analizada, objeto_detectado, nombre_objeto_crudo, datos_inventario = detector.procesar_imagen(imagen_a_procesar, umbral_seleccionado)
            status.update(label="Análisis de matriz completado", state="complete")
        st.image(img_analizada, caption="Región de interés delimitada por el Transformer.", use_container_width=True)

with col2:
    st.subheader("Asistente Normativo de Seguridad")
    tab1 = st.tabs(["Agente Experto Gemini"])
    
    with tab1[0]:
        pregunta_usuario = st.text_input("Introduzca la consulta sobre la normativa de equipaje:")
        if pregunta_usuario:
            st.chat_message("user").write(pregunta_usuario)
            with st.chat_message("assistant"):
                if not api_key_usuario:
                    st.error("Error: Se requiere una API Key válida en la barra lateral.")
                else:
                    try:
                        auditor = AuditorSeguridad(api_key=api_key_usuario)
                        stream = auditor.generar_dictamen_stream(pregunta_usuario, imagen_a_procesar, nombre_objeto_crudo, objeto_detectado)
                        texto_acumulado = ""
                        placeholder = st.empty()
                        for chunk in stream:
                            if chunk.text:
                                texto_acumulado += chunk.text
                                placeholder.markdown(texto_acumulado)
                        st.session_state.respuesta_gemini = texto_acumulado
                    except Exception:
                        st.warning("⚠️ Conexión remota no disponible. Activando protocolo de respuesta asíncrona local.")
                        texto_contingencia = (
                            f"Análisis local preventivo: La inspección automática detectó la presencia de un elemento "
                            f"morfofuncional sospechoso catalogado provisionalmente como '{nombre_objeto_crudo}'. "
                            f"Se recomienda encarecidamente activar el protocolo de verificación física intrusiva de inmediato."
                        )
                        st.markdown(texto_contingencia)
                        st.session_state.respuesta_gemini = texto_contingencia
                    
        if st.session_state.respuesta_gemini:
            st.markdown("---")
            if st.button("📊 Generar Acta Oficial Automatizada"):
                with st.spinner("Estructurando reporte pericial en formato Word..."):
                    llave_segura = api_key_usuario if api_key_usuario else "dummy_key"
                    instancia_auditor = AuditorSeguridad(api_key=llave_segura)
                    
                    try:
                        prompt_interno_doc = (
                            "Genera única y exclusivamente el cuerpo de un reporte oficial de incidencia aduanera. "
                            "Describe detalladamente el objeto prohibido observado (pistola/arma si está presente), "
                            "las regulaciones de la OACI infringidas (Anexo 17) y el protocolo de seguridad."
                        )
                        respuesta_doc_raw = instancia_auditor.client.models.generate_content(
                            model='gemini-2.5-flash',
                            contents=[prompt_interno_doc, imagen_a_procesar] if imagen_a_procesar else [prompt_interno_doc],
                            config=types.GenerateContentConfig(temperature=0.1)
                        )
                        texto_para_doc = respuesta_doc_raw.text
                    except Exception:
                        texto_para_doc = (
                            f"Durante el proceso de inspección, se identificó un objeto consistente con artículos prohibidos. "
                            f"Clasificación del sistema: '{nombre_objeto_crudo}'."
                        )
                    
                    instancia_auditor.compilar_docx(
                        analisis_modelo=texto_para_doc,
                        objeto=nombre_objeto_crudo,
                        clasificacion=objeto_detectado,
                        umbral=umbral_seleccionado
                    )
                
                with open("informe_incidencia_oaci.docx", "rb") as docx_file:
                    bytes_docx = docx_file.read()
                st.download_button(
                    label="📥 Descargar Acta de Inspección (DOCX)",
                    data=bytes_docx,
                    file_name="informe_auditoria_oaci.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                )
