import os
from typing import List, Tuple, Dict, Any

# LangChain Imports
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage, AIMessage

class RAGEngine:
    def __init__(self, persist_directory: str = "data/vector_store"):
        self.persist_directory = persist_directory
        
        # 1. Configuración de Embeddings (Debe ser IDÉNTICO a la fase de ingesta)
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # 2. Configuración del LLM Local (Ollama)
        # temperature=0 para respuestas fácticas y precisas (menos creatividad, más rigor).
        self.llm = ChatOllama(
            model="llama3", 
            temperature=0,
            keep_alive="5m"  # Mantiene el modelo en RAM 5 mins
        )
        
        # 3. Cargar Base Vectorial
        if not os.path.exists(persist_directory):
            raise FileNotFoundError(f"No se encontró la DB en {persist_directory}. Ejecuta la ingesta primero.")
            
        self.vector_db = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embedding_model
        )
        
        # 4. Configurar el Retriever
        # k=3: Recuperamos los 3 fragmentos más relevantes para darle contexto al LLM.
        self.retriever = self.vector_db.as_retriever(search_kwargs={"k": 3})
        
        # 5. Inicializar la Cadena de Conversación
        self.conversation_chain = self._setup_chain()

    def _setup_chain(self):
        """
        Configura la cadena RAG con memoria y re-escritura de preguntas.
        """
        
        # --- SUB-CADENA 1: Contextualización ---
        # Si el usuario dice "¿Cuánto cuesta?", el modelo necesita saber de qué hablamos antes.
        # Esta cadena reescribe la pregunta usando el historial.
        context_q_system_prompt = """
        Dado un historial de chat y la última pregunta del usuario, 
        formulala como una pregunta independiente que pueda entenderse sin el historial. 
        NO respondas a la pregunta, solo reformúlala si es necesario o devolvela tal cual.
        """
        context_q_prompt = ChatPromptTemplate.from_messages([
            ("system", context_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        
        history_aware_retriever = create_history_aware_retriever(
            self.llm, self.retriever, context_q_prompt
        )

        # --- SUB-CADENA 2: Respuesta y Citas ---
        qa_system_prompt = """
        Sos un asistente de IA experto en análisis documental. 
        Usa los siguientes fragmentos de contexto recuperado para responder la pregunta. 
        
        IMPORTANTE:
        1. Si no sabes la respuesta basándote en el contexto, deci que no la sabes.
        2. Mantene la respuesta concisa y profesional.
        3. SIEMPRE cita la fuente al final de tu respuesta indicando el nombre del archivo y la página.
        
        Contexto:
        {context}
        """
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        
        question_answer_chain = create_stuff_documents_chain(self.llm, qa_prompt)
        
        # Unimos todo: Historial -> Retriever -> Documentos -> LLM -> Respuesta
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
        
        return rag_chain

    def chat(self, query: str, chat_history: List[Tuple[str, str]] = []) -> Dict[str, Any]:
        """
        Procesa la pregunta y devuelve respuesta + fuentes.
        """
        # Convertimos el historial de tuplas al formato de LangChain
        langchain_history = []
        for human, ai in chat_history:
            langchain_history.append(HumanMessage(content=human))
            langchain_history.append(AIMessage(content=ai))

        # Ejecutamos la cadena
        response = self.conversation_chain.invoke({
            "input": query,
            "chat_history": langchain_history
        })
        
        # Procesamos las fuentes para mostrarlas limpio en el UI
        sources = []
        if "context" in response:
            for doc in response["context"]:
                sources.append({
                    "source": doc.metadata.get("source", "Desconocido"),
                    "page": doc.metadata.get("page", "N/A"),
                    "content_snippet": doc.page_content[:100] + "..." # Snippet para debug
                })

        return {
            "answer": response["answer"],
            "sources": sources
        }
