import os
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

class DocumentProcessor:
    def __init__(self):
        # Configuramos el modelo de Embeddings.
        # all-MiniLM-L6-v2 es rápido y eficiente para CPU.
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # Configuración del Splitter
        # chunk_size=1000: Tamaño suficiente para contener párrafos completos.
        # chunk_overlap=200: Vital para mantener contexto entre cortes.
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )

    def process_pdf(self, source_path: str, persist_directory: str) -> bool:
        """
        Carga un PDF, lo divide en chunks y lo guarda en la base vectorial.
        """
        if not os.path.exists(source_path):
            raise FileNotFoundError(f"El archivo {source_path} no existe.")

        print(f"🔄 Procesando archivo: {source_path}...")

        # 1. Carga del documento (Mantiene metadatos: source y page)
        loader = PyPDFLoader(source_path)
        documents = loader.load()
        print(f"   📄 Páginas cargadas: {len(documents)}")

        # 2. División en chunks (Splitting)
        chunks = self.text_splitter.split_documents(documents)
        print(f"   ✂️ Chunks generados: {len(chunks)}")

        # 3. Guardado en Vector Store (ChromaDB)
        # Esto crea los embeddings y los guarda localmente.
        if chunks:
            vector_db = Chroma.from_documents(
                documents=chunks,
                embedding=self.embedding_model,
                persist_directory=persist_directory
            )
            print("Documento vectorizado y guardado con éxito.")
            return True
        else:
            print("No se pudo extraer texto para generar chunks.")
            return False
