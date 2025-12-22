from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import os
import tempfile
from src.ingestion import DocumentProcessor
from src.ragengine import RAGEngine

app = Flask(__name__)
CORS(app)  # Permite peticiones desde otros orígenes si se quisiera desacoplar por completo el frontend del backend

# --- ESTADO GLOBAL ---
# En producción usaríamos una base de datos para sesiones, pero para este MVP mantenemos el motor en memoria.
engine = None

def get_engine():
    """Singleton pattern para cargar el motor solo si es necesario"""
    global engine
    if engine is None:
        if os.path.exists("data/vector_store"):
            engine = RAGEngine()
    return engine

# --- RUTAS ---

@app.route('/')
def home():
    """Sirve la interfaz gráfica (HTML)"""
    return render_template('index.html')

@app.route('/api/upload', methods=['POST'])
def upload_files():
    """Endpoint para subir y procesar PDFs"""
    global engine
    
    if 'files' not in request.files:
        return jsonify({"error": "No se enviaron archivos"}), 400
    
    files = request.files.getlist('files')
    processor = DocumentProcessor()
    
    # Directorio temporal para procesamiento seguro
    with tempfile.TemporaryDirectory() as temp_dir:
        processed_count = 0
        for file in files:
            if file.filename == '':
                continue
            
            # Guardar temporalmente
            file_path = os.path.join(temp_dir, file.filename)
            file.save(file_path)
            
            # Procesar
            success = processor.process_pdf(file_path, "data/vector_store")
            if success:
                processed_count += 1
    
    # Recargar el motor para que vea los nuevos datos
    engine = RAGEngine()
    
    return jsonify({
        "message": f"Procesamiento completado. {processed_count} archivos vectorizados.",
        "status": "success"
    })

@app.route('/api/chat', methods=['POST'])
def chat():
    """Endpoint principal de conversación"""
    engine_instance = get_engine()
    
    if not engine_instance:
        return jsonify({"error": "El motor no está listo. Sube documentos primero."}), 503

    data = request.json
    user_query = data.get('message')
    history = data.get('history', []) # El frontend nos manda el historial
    
    # Convertir historial de JSON (lista de dicts) a Tuplas para el RAG Engine
    # Formato esperado por rag_engine: [("Hola", "Hola soy AI"), ...]
    tuple_history = [(h['user'], h['bot']) for h in history]

    try:
        response = engine_instance.chat(user_query, tuple_history)
        return jsonify(response) # Devuelve {answer: "...", sources: [...]}
    except Exception as e:
        print(f"Error en chat: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # debug=True permite que el servidor se reinicie si se cambia el código
    app.run(debug=True, port=5000)