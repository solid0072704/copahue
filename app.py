import streamlit as st
import requests
import pandas as pd
import re
import io 

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="Reportes Forpay", layout="wide")

def main():
    st.title("📊 Generador de Reportes Forpay")
    st.markdown("Selecciona la empresa y el estado de la cuota para descargar el reporte detallado.")
    st.info("ℹ️ Nota: Para elegir la carpeta de destino, configura tu navegador para que 'Pregunte dónde guardar' en la configuración de Descargas.")

    # ---------------------------------------------------------
    # 1. CATÁLOGO DE EMPRESAS
    # ---------------------------------------------------------
    CATALOGO_EMPRESAS = {
        "Boulevard Macul SpA": "e0e05fd9bf6d7faa8771089cff195f2a",
        "La Oración SpA": "013416e2220558709345126395c724e4",
        "Irarrázaval 2137 SpA": "d4e87c5a84d852ca256ec42661a809a8",
        "Irarrázaval 4870 SpA":"df66fc4f1af5bb9003909e508f2c05e1",
        "Inmobiliaria y Constructora Costa Brava SpA":"8ddf83c78c29c71e8d2df8998dfe2216",
        "Rentas Inmobiliarias Solaro SpA":"6f12701c16f597574c356dcd061c360c",
        "Hipodromo":"672d6dbda1515e2a06ae50dae26db3b5",
        "LOS BARBECHOS SpA":"3b76399bfe74059f2d1ddaf12fabcbfc"
    }

    with st.sidebar:
        st.header("Configuración")
        
        # Selección de Empresa
        nombre_empresa = st.selectbox(
            "Seleccione Empresa:",
            options=list(CATALOGO_EMPRESAS.keys())
        )
        
        token_seleccionado = CATALOGO_EMPRESAS[nombre_empresa]
        st.success(f"Empresa seleccionada: {nombre_empresa}")

        st.divider()

        # Selección de Estado
        opciones_estado = [
            "Pendiente-Moroso",
            "Pendiente-No-Moroso",
            "Pendiente-Hoy",
            "Pendiente",
            "Pagada",
            "Anulada",
            "En Proceso de Cobro",
            "Repactada",
            "Finalizada",
            "Reversada"
        ]
        
        estado_seleccionado = st.selectbox(
            "Estado de Cuota:",
            options=opciones_estado,
            index=0 
        )

    # ---------------------------------------------------------
    # 2. BOTÓN DE ACCIÓN
    # ---------------------------------------------------------
    # Usamos un contenedor para separar la lógica de UI
    if st.button("🔍 Buscar y Preparar Archivo", type="primary"):
        with st.spinner('Conectando a la API y procesando datos...'):
            df_resultado = procesar_datos(token_seleccionado, estado_seleccionado, nombre_empresa)
            
            if df_resultado is not None:
                generar_descarga(df_resultado, nombre_empresa)

def procesar_datos(token, estado_cuota, nombre_empresa_filtro):
    # Configuración API
    url = "https://t.forpayservices.cl/api-empresas-obtener-cuotas-v5"
    params = {
        "token": token,
        "estado_cuota": estado_cuota,
        "estado_compromiso": "Activa"
    }

    # 1. Conexión
    try:
        response = requests.get(url, params=params, timeout=30) # Agregado timeout
        response.raise_for_status()
        data_json = response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error de conexión con la API: {e}")
        return None
    except ValueError:
        st.error("La respuesta de la API no es un JSON válido.")
        return None

    # 2. Procesamiento
    try:
        # Normalización inicial
        if isinstance(data_json, dict) and 'data' in data_json:
            df = pd.DataFrame(data_json['data'])
        elif isinstance(data_json, list):
            df = pd.DataFrame(data_json)
        else:
            df = pd.DataFrame([data_json])

        if df.empty:
             st.warning(f"La API respondió correctamente pero no trajo datos.")
             return None

        # A) Desempaquetar la columna raíz 'data'
        if 'data' in df.columns:
            # Validar que la columna data tenga contenido parseable
            if not df.empty and isinstance(df['data'].iloc[0], dict):
                columna_expandida = df['data'].apply(pd.Series)
                df = df.drop('data', axis=1)
                
                # Corrección duplicados Nivel 1
                cols_existentes = set(df.columns)
                cols_nuevas = set(columna_expandida.columns)
                duplicadas = cols_existentes.intersection(cols_nuevas)
                
                if duplicadas:
                    rename_dict = {col: f"{col}_detalle" for col in duplicadas}
                    columna_expandida = columna_expandida.rename(columns=rename_dict)
                
                df = pd.concat([df, columna_expandida], axis=1)

        # B) Expandir la columna 'cuotas'
        if 'cuotas' in df.columns:
            df = df.explode('cuotas')
            df = df.reset_index(drop=True)
            
            cuotas_expandidas = df['cuotas'].apply(lambda x: pd.Series(x) if isinstance(x, dict) else pd.Series())
            
            # Corrección duplicados Nivel 2 (Cuotas vs Principal)
            cols_df = set(df.columns)
            cols_cuotas = set(cuotas_expandidas.columns)
            interseccion = cols_df.intersection(cols_cuotas)
            
            if interseccion:
                mapeo_renombre = {col: f"{col}_cuota" for col in interseccion}
                cuotas_expandidas = cuotas_expandidas.rename(columns=mapeo_renombre)

            df = df.drop('cuotas', axis=1)
            df = pd.concat([df, cuotas_expandidas], axis=1)

        df = df.fillna("-")

        # Filtro de seguridad
        if 'empresa' in df.columns:
            filtro = df['empresa'].astype(str).str.contains(nombre_empresa_filtro, case=False, na=False)
            df = df[filtro]

        if df.empty:
            st.warning(f"No se encontraron registros para '{nombre_empresa_filtro}' con estado '{estado_cuota}' después del filtrado.")
            return None

        return df

    except Exception as e:
        st.error(f"Ocurrió un error al procesar la estructura de datos: {e}")
        st.exception(e)
        return None

def generar_descarga(df, nombre_empresa_filtro):
    # 3. Mostrar vista previa
    st.success(f"¡Éxito! Se encontraron {len(df)} registros procesados.")
    st.dataframe(df.head()) 

    # 4. Generar Excel
    buffer = io.BytesIO()
    
    try:
        # Usamos xlsxwriter como motor
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='Reporte')
            # Ajuste automático de columnas (opcional pero estético)
            worksheet = writer.sheets['Reporte']
            for i, col in enumerate(df.columns):
                worksheet.set_column(i, i, 20)
                
        buffer.seek(0)

        # Nombre del archivo
        nombre_limpio = re.sub(r'[\\/*?:"<>|]', "", nombre_empresa_filtro)
        nombre_archivo = f"Morosos forpay {nombre_limpio}.xlsx"

        # Botón de descarga
        st.download_button(
            label="📥 Descargar Archivo Excel",
            data=buffer,
            file_name=nombre_archivo,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    
    except Exception as e:
        st.error(f"Error generando el archivo Excel: {e}")

if __name__ == "__main__":
    main()