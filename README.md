# Private Doc Intel - RAG System (pdf_Searcher)

Un asistente conversacional impulsado por IA diseñado para interactuar con documentos PDF privados. Esta aplicación implementa una arquitectura **RAG (Retrieval-Augmented Generation)** completa, permitiendo la ingesta de documentos, su vectorización local y la posterior consulta en lenguaje natural basándose estrictamente en el contexto proporcionado.

## Características Principales
* **Ingesta Documental:** Procesamiento dinámico de archivos PDF utilizando particionado inteligente de texto para preservar la coherencia semántica.
* **Base de Datos Vectorial Local:** Implementación de `ChromaDB` para el almacenamiento persistente y búsqueda de similitud.
* **Generación de Embeddings Eficiente:** Uso del modelo `all-MiniLM-L6-v2` (Sentence-Transformers) optimizado para un rendimiento ágil.
* **Utilizacion de Groq:** Integración con la API de **Groq** para la generación de respuestas.

## Stack Tecnológico
* **Backend:** Python, Flask, FastAPI
* **IA & Orquestación:** LangChain, Hugging Face Embeddings
* **Base de Datos Vectorial:** ChromaDB
* **LLM Provider:** Groq API (Llama 3)
* **Frontend:** HTML5, Vanilla JS, Bootstrap 5



## Estructura del Proyecto

```text
pdf_Searcher/
├── data/
│   └── vector_store/       # Almacenamiento persistente de ChromaDB (SQLite)
├── src/                    
│   ├── ingestion.py        # Pipeline ETL: Extraccion, Transformacion, Carga
│   └── ragengine.py        # Motor RAG: Retriever, prompts y conexión con Groq LLM
├── static/                 # Assets estáticos (CSS/JS)
├── templates/              
│   └── index.html          # Interfaz de usuario principal
├── .env.example            # Plantilla de variables de entorno
├── requirements.txt        # Dependencias del proyecto
└── server.py               # Punto de entrada de la API web
```
## Instalación y Uso Local
1) ### Clonar el repositorio
2) ### Crear y activar un entorno virtual:
   python -m venv venv (Linux), .\venv\Scripts\activate (Windows)
3) ### Instalar dependencias:
   pip install -r requirements.txt
4) ### Configurar variables de entorno:
   Crear un archivo .env en la raíz del proyecto (ejemplo en el archivo .env.example) y agregar tu clave de Groq
5) ### Iniciar el servidor:
   python server.py


