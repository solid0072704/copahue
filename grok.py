import os
import streamlit as st 
# --- NUEVAS IMPORTACIONES ---
from dotenv import load_dotenv 
# ---------------------------
# --- IMPORTACIONES DE LLAMA_INDEX ---
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    Settings,
    PromptTemplate,
    StorageContext,
    load_index_from_storage
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.groq import Groq
# ------------------------------------

# --- CARGAR VARIABLES DE ENTORNO ---
# Esto lee tu archivo .env y carga la clave en os.environ
load_dotenv()
# -----------------------------------

# --- CONFIGURACIÓN INICIAL ---
RUTA_DE_TUS_DOCUMENTOS = "./PDF" 
EMBEDDING_MODEL = "mxbai-embed-large" 
RUTA_PERSISTENCIA = "./storage_vectorial" 
# ----------------------------------------


# --- INTERFAZ PRINCIPAL DE STREAMLIT ---
st.title("🤖 Chat con tus Documentos (con Groq 🚀)")
st.caption("Esta app usa Groq para respuestas ultra-rápidas.")

# --- BARRA LATERAL (SIDEBAR) PARA CONFIGURACIÓN ---
with st.sidebar:
    
    # Logo
    try:
        st.image("logo.png", width=150) 
    except FileNotFoundError:
        st.warning("No se encontró 'logo.png'.")

    st.title("Configuración")
    st.markdown("Ajusta los parámetros antes de iniciar el chat.")

    # Botón de Refresco
    if st.button("🔄 Refrescar Listas (Archivos/Modelos)"):
        st.cache_data.clear()
        st.rerun()

    # --- 1. LÓGICA DE SELECCIÓN DE ARCHIVOS ---
    st.divider()
    st.subheader("1. Selección de Archivo")
    
    @st.cache_data
    def escanear_archivos(ruta_base):
        archivos_unicos = set()
        try:
            for root, _, files in os.walk(ruta_base):
                for file in files:
                    if file.lower().endswith(".pdf"):
                        archivos_unicos.add(os.path.join(root, file))
        except Exception as e:
            st.error(f"Error al escanear la carpeta: {e}")
            return []
        
        lista_archivos = [
            (os.path.basename(ruta), ruta) for ruta in sorted(list(archivos_unicos))
        ]
        return lista_archivos

    with st.spinner("Escaneando documentos..."):
        lista_archivos_tuplas = escanear_archivos(RUTA_DE_TUS_DOCUMENTOS)
        
    opciones_archivos = {
        "Todos los archivos": RUTA_DE_TUS_DOCUMENTOS,
        **{nombre: ruta for nombre, ruta in lista_archivos_tuplas}
    }

    archivo_seleccionado_nombre = st.selectbox(
        "Elige un archivo (o todos):",
        options=opciones_archivos.keys()
    )
    ruta_a_cargar = opciones_archivos[archivo_seleccionado_nombre]

    # --- 2. LÓGICA DE SELECCIÓN DE LLM (Groq) ---
    st.divider()
    st.subheader("2. Selección de LLM (Groq)")

    modelos_disponibles = [
        "llama3-8b-8192",      
        "mixtral-8x7b-32768", 
        "gemma-7b-it"         
    ]

    LLM_MODEL = st.selectbox(
        "Elige un modelo LLM de Groq:",
        options=modelos_disponibles,
        index=0 
    )

    # --- 3. CONFIGURACIÓN DE CHUNKS ---
    st.divider()
    st.subheader("3. Configuración de Chunks")
    chunk_size = st.number_input("Tamaño del chunk (Chunk Size):", min_value=100, max_value=8000, value=1000, step=100)
    chunk_overlap = st.number_input("Solapamiento (Chunk Overlap):", min_value=0, max_value=1000, value=200, step=50)


# --- LÓGICA PRINCIPAL (CARGA Y CREACIÓN DEL ÍNDICE) ---
@st.cache_resource(show_spinner="Cargando o creando el índice...")
def cargar_y_crear_indice(_ruta_a_cargar, _llm_model, _chunk_size, _chunk_overlap):
    
    # 1. Crear un nombre de directorio único
    nombre_base_limpio = os.path.basename(_ruta_a_cargar).replace('.', '_').replace(' ', '')
    
    directorio_persistencia = os.path.join(
        RUTA_PERSISTENCIA, 
        f"{nombre_base_limpio}_chunk{_chunk_size}_overlap{_chunk_overlap}"
    )
    
    # --- INICIO DE LA MODIFICACIÓN (.env) ---
    
    # 2. Configurar el LLM (Groq) y Embeddings (Ollama)
    
    # Leemos la API key desde las variables de entorno (cargadas de .env)
    groq_api_key = os.environ.get("GROQ_API_KEY")

    # Comprobamos que la variable exista
    if not groq_api_key:
        st.error("No se encontró GROQ_API_KEY. Por favor, añádela a tu archivo .env")
        return None
        
    st.write(f"Conectando a Groq (Modelo: {_llm_model})...")
    Settings.llm = Groq(
        model=_llm_model, 
        api_key=groq_api_key # <-- USAMOS LA VARIABLE
    )
    
    # --- FIN DE LA MODIFICACIÓN ---

    try:
        st.write(f"Cargando Embedding Model: {EMBEDDING_MODEL} (desde Ollama)")
        Settings.embed_model = OllamaEmbedding(
            model_name=EMBEDDING_MODEL
        )
    except Exception as e:
        st.error(f"Error al conectar con Ollama (para Embeddings): {e}. ¿Está corriendo?")
        return None

    Settings.text_splitter = SentenceSplitter(
        chunk_size=_chunk_size, 
        chunk_overlap=_chunk_overlap
    )

    # 3. Comprobar si el índice ya existe en disco
    if not os.path.exists(directorio_persistencia):
        st.info("Índice no encontrado. Creando y guardando uno nuevo... (Esto puede tardar)")
        
        if os.path.isfile(_ruta_a_cargar):
            loader = SimpleDirectoryReader(input_files=[_ruta_a_cargar])
        else:
            loader = SimpleDirectoryReader(input_dir=_ruta_a_cargar, recursive=True)
            
        try:
            documents = loader.load_data(show_progress=True)
            if not documents:
                st.error("No se pudieron cargar documentos. Revisa la ruta.")
                return None
        except Exception as e:
            st.error(f"Error crítico al cargar documentos: {e}")
            return None

        index = VectorStoreIndex.from_documents(
            documents,
            show_progress=True
        )
        
        st.write(f"Guardando índice en: {directorio_persistencia}")
        index.storage_context.persist(persist_dir=directorio_persistencia)
        st.success("¡Índice creado y guardado!")

    else:
        st.info(f"Cargando índice existente desde: {directorio_persistencia}")
        storage_context = StorageContext.from_defaults(persist_dir=directorio_persistencia)
        index = load_index_from_storage(storage_context)
        st.success("¡Índice cargado desde disco!")
    

    # 4. Definir el Prompt Template
    prompt_template_str = """
    Eres un asistente de IA que SOLO habla español. Tu única tarea es responder en español.
    Responde la pregunta del usuario basándote estricta y únicamente en el contexto proporcionado.
    Si la respuesta no se encuentra en el contexto, debes decir: "No tengo información sobre eso".
    Bajo NINGUNA circunstancia respondas en inglés.

    Contexto:
    <context>
    {context_str}
    </context>

    Pregunta: {query_str}

    Respuesta (en español):
    """
    qa_template = PromptTemplate(prompt_template_str)

    # 5. Crear el Motor de Consulta y devolverlo
    query_engine = index.as_query_engine(
        text_qa_template=qa_template,
        similarity_top_k=3 
    )
    return query_engine

# --- FIN DE LA FUNCIÓN DE CACHÉ ---


# Inicializamos query_engine como None ANTES del try
query_engine = None 

try:
    query_engine = cargar_y_crear_indice(
        ruta_a_cargar,
        LLM_MODEL,
        chunk_size,
        chunk_overlap
    )
except Exception as e:
    st.error(f"Error al inicializar el motor de consulta: {e}")
    
# Esta comprobación ahora funciona de forma segura
if query_engine is None:
    st.error("No se pudo inicializar el motor de consulta. Revisa la configuración y los documentos.")
    st.stop()


# --- LÓGICA DE LA INTERFAZ DE CHAT ---

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "¡Hola! Estoy listo para responder preguntas sobre tus documentos."}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Escribe tu pregunta aquí..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty() 
        
        with st.spinner("Pensando... (Usando Groq 🚀)"):
            try:
                response = query_engine.query(prompt)
                full_response = str(response)
                
            except Exception as e:
                full_response = f"Error al procesar la consulta: {e}"
        
        message_placeholder.markdown(full_response)
    
    st.session_state.messages.append({"role": "assistant", "content": full_response})