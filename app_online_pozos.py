import os
import time
import json
import logging
import pandas as pd
from flask import Flask, request, jsonify
from google.cloud import storage, bigquery, tasks_v2
from vertexai import init
from vertexai.preview.generative_models import GenerativeModel, Part
import vertexai.preview.generative_models as generative_models

# Config global
GCP_PROJECT = "extrac-datos-geosys-production"
GCP_REGION = "us-central1"
CLOUD_TASK_QUEUE = "online-pozos"
WORKER_URL = "https://online-pozos-386277896892.us-central1.run.app/process_single"

# Inicializar Vertex AI y logging
init(project=GCP_PROJECT, location="us-central1")
logging.basicConfig(level=logging.INFO)

# Listas personalizadas
operadora = pd.read_csv("operadora.csv").iloc[:, 0].dropna().unique().tolist()
logname = pd.read_csv("Logname.csv").iloc[:, 0].dropna().unique().tolist()
contratista = pd.read_csv("contratista.csv").iloc[:, 0].dropna().unique().tolist()
curvas = pd.read_csv("Curvas.csv").iloc[:, 0].dropna().unique().tolist()

# Prompt base (omitido por brevedad)
prompt_base = """
Eres un especialista en extracción de entidades de documentos. Dado un documento, tu tarea es extraer el valor de texto de las siguientes entidades:

                            {{
                                "LOGNAME": "",
                                "COMPANY": "",
                                "WELL": "",
                                "FIELD": "",
                                "COUNTRY": "",
                                "LOG_MEASURED_FROM": "",
                                "KB": "",
                                "DF": "",
                                "GL": "",
                                "RUN_N°": "",
                                "DATE": "",
                                "FIRST_READING": "",
                                "LAST_READING": "",
                                "DEPTH_REACHED": "",
                                "BOTTOM_DRILLER": "",
                                "MUD_NATURE": "",
                                "DENSITY": "",
                                "VISCOSITY": "",
                                "MUD_RESISTIVITY": "",
                                "MUD_RESISTIVITY_BHT": "",
                                "PH": "",
                                "FLUID_LOSS": "",
                                "RMF": "",
                                "RMC": "",
                                "CURVAS": [],
                                "CODIGO": "",
                                "CONTRATANTE": "",
                                "STATE": "",
                                "SPUD_DATE_COMENZÓ_A_PERFORAR": "",
                                "COMPLETED": "",
                                "TOTAL_DEPTH": ""
                            }}


                            - 'LOGNAME':
                                Busca el LOGNAME dentro de la franja oscura ubicada en la parte superior del registro eléctrico (electrical log). Sigue estas reglas:

                                Primera prioridad: Verifica si contiene la palabra "log" o coincide con un elemento de la lista proporcionada: {logname_list}.
                                Excepciones: En caso de que no contenga "log", identifica nombres comunes de registros como:
                                HOLE SECTION SURVEY
                                FIELD PRINT
                                COMPENSATED NEUTRON - FORMATION DENSITY
                                Cualquier otro nombre relevante que no contenga "log" pero que pueda considerarse parte del registro eléctrico.
                                Obligatoriedad: Es obligatorio extraer un valor, por lo que si no identificas un valor directamente, revisa nuevamente la franja oscura para encontrar el registro más probable relacionado.
                                Ignora valores que no estén dentro de la franja oscura o que no sean parte del registro eléctrico.

                            - 'COMPANY': Extrae el valor del campo "Compañía" siguiendo estas reglas:
                                Comparación con la lista:
                                Guíate exclusivamente de los valores proporcionados en la lista {compañia_list} para identificar el nombre de la compañía.
                                Busca coincidencias exactas o parciales entre los nombres en los logos y los elementos de la lista.
                                Uso de logos:
                                Identifica los nombres o textos que se encuentren en los logos presentes en el documento o imagen.
                                Compara estos textos con los valores de {compañia_list} para encontrar la mejor coincidencia.
                                Resultado final:
                                Si encuentras una coincidencia válida en la lista, extrae ese nombre como el valor de "Compañía".
                                Si no encuentras un valor válido (ni en los logos ni en la lista), deja el campo como null.

                            - 'WELL': Extrae el valor del campo "Pozo". Si no encuentras el valor, déjalo en null.
                            - 'FIELD': Extrae el valor del campo "Campo". Si no encuentras el valor, déjalo en null.
                            - 'COUNTRY': Extrae el valor del campo "País". Si no encuentras el valor, déjalo en null.
                            - 'LOG_MEASURED_FROM': Busca la profundidad de referencia etiquetada como DF, KB o GL. Si no encuentras el valor, déjalo en null.
                            
                            - 'KB': 
                                Identifica y extrae el valor de KB o K.B desde la sección "Elevation".
                                Si no encuentras un valor específico, marca el campo como null.
                            - 'DF': 
                                identifica y extrae el valor asociado a DF o D.F desde la sección "Elevation".
                                Si no encuentras un valor específico, marca el campo como null.
                            
                            - 'GL': 
                                Identifica y extrae el valor correspondiente a GL desde la sección "Elevation".
                                Si no encuentras un valor específico, marca el campo como null.
                            
                            - 'RUN N°': Extrae el valor del número de corrida. Si no encuentras el valor, déjalo en null.
                            - 'DATE': Extrae el valor de la fecha y conviértelo al formato YYYY-MM-DD. Si no aparece el día, usa "01" como día. Si no encuentras el valor, déjalo en null.
                            - 'FIRST_READING': Extrae el valor de la primera lectura. Si no encuentras el valor, déjalo en null.
                            - 'LAST_READING': Extrae el valor de la última lectura. Si no encuentras el valor, déjalo en null.
                            - 'DEPTH_REACHED': Extrae el valor de la profundidad alcanzada. Si no encuentras el valor, déjalo en null.
                            - 'BOTTOM_DRILLER': Extrae el valor del diámetro del trépano en el fondo. Si no encuentras el valor, déjalo en null.
                            - 'MUD_NATURE': Extrae el valor del tipo de lodo. Si no encuentras el valor, déjalo en null.
                            - 'DENSITY': Extrae el valor de la densidad del lodo. Si no encuentras el valor, déjalo en null.
                            - 'VISCOSITY': Extrae el valor de la viscosidad del lodo. Si no encuentras el valor, déjalo en null.
                            - 'MUD_RESISTIVITY': Extrae el valor de la resistividad del lodo. Si no encuentras el valor, déjalo en null.
                            - 'MUD_RESISTIVITY_BHT': Extrae el valor de la resistividad del lodo a temperatura de fondo. Si no encuentras el valor, déjalo en null.
                            - 'PH': Extrae el valor del Ph del lodo. Si no encuentras el valor, déjalo en null.
                            - 'FLUID_LOSS': Extrae el valor de la pérdida de fluido. Si no encuentras el valor, déjalo en null.
                            
                            - 'RMF': Extrae el valor de la resistividad del filtrado de lodo. Puede venir en varias formas ej: Rmf,R(mf) at Surface,etc. Si no encuentras el valor, déjalo en null.
                            - 'RMC': Extrae el valor de la resistividad del revoque de lodo. Puede venir en varias formas ej: Rmc,R(mc) at Surface. Si no encuentras el valor, déjalo en null.

                            - 'CURVAS': 
                            Extrae los datos relacionados con las curvas y los datos litológicos desde los documentos proporcionados. Sigue estas instrucciones de manera estricta:
                            Curvas Principales:
                            Identifica las curvas principales presentes en los documentos. Si aparecen etiquetas como:
                            Resistivity
                            Gamma Ray (GR)
                            Neutron Porosity
                            Density (Bulk or Matrix)
                            Spontaneous Potential (SP)
                            Incluye también subcategorías específicas (por ejemplo, "Detector Cercano", "Detector Lejano", "C1", "C2", etc.).
                        
                            Datos Litológicos:

                            Identifica las palabras clave relacionadas con:
                            Litología (Lithology descriptions, Sample descriptions, Fluorescencia, etc.).
                            Cromatografía (C1, C2, C3, Total Gas, etc.).
                            Drilling Rate (Promedio de perforación).
                          
                            Estructura General:

                            Si no encuentras un dato, indica su ausencia con null.
                            Considera que algunos documentos podrían usar términos alternativos para las curvas o datos litológicos. Incluye todas las variantes relevantes que puedas detectar.
                            Prioriza Datos en Tablas y Gráficos:

                            Identifica información presente en secciones tabulares o gráficas (como las incluidas en las imágenes proporcionadas) y extrae datos clave como:
                            Rangos de medición (por ejemplo, "0-150 API" para GR, "0-15 ohm-m" para resistividad).
                            Unidades relevantes (lb, ft/min, cps, ppm).

                            Quiero que concatenes todos los datos relacionado con las curvas y los datos litológicos separados con comas. Ejemplo:

                            "Curvas": "Gamma Ray, Density, Cromatografía, Drilling Rate, Litología, Fluorescencia"
                           

                            - 'CODIGO': Extrae el código de la etiqueta. Si no encuentras el valor, déjalo en null.

                            - 'CONTRATANTE': Extrae el valor del contratante, identificado como logo o texto en la parte superior izquierda o superior derecha o en la franja de Logname. Guíate de esta lista:
                            {contratista_list}

                            - 'STATE': Extrae el valor del departamento o estado. Si no encuentras el valor, déjalo en null.
                            - SPUD_DATE_COMENZÓ_A_PERFORAR: Extrae la fecha de inicio de perforación y conviértela al formato YYYY-MM-DD. Si no encuentras el valor, déjalo en null.
                            - 'COMPLETED': Extrae la fecha de finalización de perforación y conviértela al formato YYYY-MM-DD. Si no encuentras el valor, déjalo en null.
                            - 'TOTAL_DEPTH': Extrae el valor de la profundidad total. Si no encuentras el valor, déjalo en null.

                            Devuelve el resultado en el siguiente formato estructurado:

                            Devuelve los resultados organizados en el siguiente ejemplo formato JSON (solo de esta manera sin comentarios adicionales, solo el formato):
                            {{
                                "LOGNAME": "Gamma Ray Log",
                                "COMPANY": "EnergyCorp Ltd.",
                                "WELL": "EC-45",
                                "FIELD": "Campo Norte",
                                "COUNTRY": "Argentina",
                                "LOG_MEASURED_FROM": "100 m",
                                "KB": "35 m",
                                "DF": "30 m",
                                "GL": "25 m",
                                "RUN_N°": 3,
                                "DATE": "2024-06-20",
                                "FIRST_READING": "100 m",
                                "LAST_READING": "2000 m",
                                "DEPTH_REACHED": "2000 m",
                                "BOTTOM_DRILLER": "John Doe Drillers",
                                "MUD_NATURE": "Aceite",
                                "DENSITY": "1.2 g/cm³",
                                "VISCOSITY": "45 cP",
                                "MUD_RESISTIVITY": "0.15 ohm·m",
                                "MUD_RESISTIVITY_BHT": "0.10 ohm·m",
                                "PH": 8.5,
                                "FLUID_LOSS": "12 ml",
                                "RMF": "0.25 ohm·m",
                                "RMC": "0.20 ohm·m",
                                "CURVAS": "Gamma Ray, Resistividad, Densidad, Porosidad",
                                "CODIGO": "WELL-7890",
                                "CONTRATANTE": "PetroServicios S.A.",
                                "STATE": "Neuquén",
                                "SPUD_DATE_COMENZÓ_A_PERFORAR": "2024-05-01",
                                "COMPLETED": "2024-06-18",
                                "TOTAL_DEPTH": "2000 m"
                            }}

                            NOTA IMPORTANTE:

                            - Analiza el archivo con lógica inferencial y patrones para detectar elementos relevantes.
                            - Prioriza precisión y evita extraer valores incorrectos por similitud.
                            - Se debe seguir el esquema JSON durante la extracción.
                            - Los valores solo deben incluir cadenas de texto encontradas en el documento.
"""  # Usa el prompt completo que ya tienes

def build_prompt():
    return prompt_base.format(
        compañia_list=operadora,
        logname_list=logname,
        contratista_list=contratista,
        curvas_list=curvas
    )

def download_blob_as_bytes(bucket_name, blob_path):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    return blob.download_as_bytes(), blob.content_type

def generate_from_document(document1, prompt, model_version):
    model = GenerativeModel(model_version)
    responses = model.generate_content(
        [document1, prompt],
        generation_config={
            "max_output_tokens": 8192,
            "temperature": 0,
            "top_p": 0.95,
        },
        safety_settings={
            generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
            generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
            generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
            generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
        },
    )
    return responses.text.strip()

def save_to_bigquery(file_name, respuesta_texto):    
    client = bigquery.Client()
    table_id = f"{client.project}.gf_pozos.resultados_pozos"

    try:
        client.get_table(table_id)
    except Exception:
        schema = [
            bigquery.SchemaField("archivo", "STRING"),
            bigquery.SchemaField("respuesta_modelo", "STRING"),
            bigquery.SchemaField("fecha_procesamiento", "TIMESTAMP"),  # NUEVO
        ]
        client.create_table(bigquery.Table(table_id, schema=schema))

    query = f"SELECT COUNT(*) as total FROM {table_id} WHERE archivo = @archivo"
    job_config = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ScalarQueryParameter("archivo", "STRING", file_name)]
    )
    existe = next(client.query(query, job_config=job_config).result()).total > 0

    if existe:
        update_query = f"""
            UPDATE {table_id} 
            SET respuesta_modelo = @respuesta, 
                fecha_procesamiento = @fecha 
            WHERE archivo = @archivo
        """
        update_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("archivo", "STRING", file_name),
                bigquery.ScalarQueryParameter("respuesta", "STRING", respuesta_texto),
                bigquery.ScalarQueryParameter("fecha", "TIMESTAMP", datetime.utcnow()),
            ]
        )
        client.query(update_query, job_config=update_config).result()
    else:
        client.insert_rows_json(table_id, [{
            "archivo": file_name, 
            "respuesta_modelo": respuesta_texto,
            "fecha_procesamiento": datetime.utcnow().isoformat()
        }])


def save_metrics_to_bigquery(file_name, status, error_msg=None, tiempo_procesamiento=None, model_version=None):
    """
    Guarda métricas de procesamiento en BigQuery
    """
    client = bigquery.Client()
    table_id = f"{client.project}.gf_pozos.metricas_procesamiento"
    
    try:
        client.get_table(table_id)
    except Exception:
        schema = [
            bigquery.SchemaField("archivo", "STRING"),
            bigquery.SchemaField("fecha_procesamiento", "TIMESTAMP"),
            bigquery.SchemaField("status", "STRING"),  # 'success' o 'error'
            bigquery.SchemaField("error_mensaje", "STRING"),
            bigquery.SchemaField("tiempo_procesamiento_seg", "FLOAT"),
            bigquery.SchemaField("model_version", "STRING"),
        ]
        client.create_table(bigquery.Table(table_id, schema=schema))
    
    row = {
        "archivo": file_name,
        "fecha_procesamiento": datetime.utcnow().isoformat(),
        "status": status,
        "error_mensaje": error_msg,
        "tiempo_procesamiento_seg": tiempo_procesamiento,
        "model_version": model_version
    }
    
    client.insert_rows_json(table_id, [row])
# Flask
app = Flask(__name__)

@app.route("/", methods=["GET"])
def health():
    return "API activa", 200

@app.route("/enqueue_tasks", methods=["POST"])
def enqueue_tasks():
    try:
        data = request.get_json()
        bucket_path = data["bucket_path"]
        cantidad = int(data.get("cantidad", 10))
        model_version = data.get("model_version", "gemini-2.5-flash")

        bucket_name, prefix = bucket_path.replace("gs://", "").split("/", 1)
        client = storage.Client()
        blobs = list(client.list_blobs(bucket_name, prefix=prefix))
        blobs = [b.name for b in blobs if b.name.endswith((".pdf", ".jpg", ".png", ".tiff", ".tif"))][:cantidad]

        task_client = tasks_v2.CloudTasksClient()
        parent = task_client.queue_path(GCP_PROJECT, GCP_REGION, CLOUD_TASK_QUEUE)

        for blob_name in blobs:
            payload = {
                "bucket_name": bucket_name,
                "blob_name": blob_name,
                "model_version": model_version
            }
            task = {
                "http_request": {
                    "http_method": tasks_v2.HttpMethod.POST,
                    "url": WORKER_URL,
                    "headers": {"Content-Type": "application/json"},
                    "body": json.dumps(payload).encode()
                }
            }
            task_client.create_task(parent=parent, task=task)

        return jsonify({"tareas_enviadas": len(blobs)}), 200
    except Exception as e:
        logging.exception("[ERROR] enqueue_tasks")
        return jsonify({"error": str(e)}), 500

@app.route("/process_single", methods=["POST"])
def process_single():
    start_time = time.time()
    data = request.get_json()
    bucket_name = data["bucket_name"]
    blob_name = data["blob_name"]
    model_version = data.get("model_version", "gemini-2.5-flash")
    
    try:
        # Procesamiento
        file_bytes, mime_type = download_blob_as_bytes(bucket_name, blob_name)
        part = Part.from_data(mime_type=mime_type, data=file_bytes)
        prompt = build_prompt()
        respuesta = generate_from_document(part, prompt, model_version)
        
        # Guardar resultado
        save_to_bigquery(blob_name, respuesta)
        
        # Guardar métricas de éxito
        tiempo_procesamiento = time.time() - start_time
        save_metrics_to_bigquery(
            file_name=blob_name,
            status="success",
            tiempo_procesamiento=tiempo_procesamiento,
            model_version=model_version
        )
        
        logging.info(f"✓ Procesado exitoso: {blob_name} en {tiempo_procesamiento:.2f}s")
        return jsonify({
            "procesado": blob_name,
            "tiempo_seg": tiempo_procesamiento
        }), 200

    except Exception as e:
        # Guardar métricas de error
        tiempo_procesamiento = time.time() - start_time
        error_msg = str(e)
        
        save_metrics_to_bigquery(
            file_name=blob_name,
            status="error",
            error_msg=error_msg,
            tiempo_procesamiento=tiempo_procesamiento,
            model_version=model_version
        )
        
        logging.exception(f"✗ Error procesando {blob_name}")
        return jsonify({
            "error": error_msg,
            "archivo": blob_name
        }), 500
      
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
