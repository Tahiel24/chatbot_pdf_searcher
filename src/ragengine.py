import os
from typing import List, Tuple, Dict, Any
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts.chat import ChatPromptTemplate
from operator import itemgetter
from langchain_core.runnables import RunnableLambda
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv() # Cargar variables de entorno desde .env

class RAGEngine:
    def __init__(self, persist_directory: str = "data/vector_store"):
        self.persist_directory = persist_directory

        # 1. Embeddings (idéntico a la ingesta)
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # 2. LLM local (Ollama)
        self.llm = ChatGroq(
            model="llama-3.3-70b-versatile",     #Cambiar esta linea segun el modelo llm que se vaya a utilizar
            temperature=0,
            api_key=os.getenv("GROQ_API_KEY")
        )

        # 3. Vector DB
        if not os.path.exists(persist_directory):
            raise FileNotFoundError(
                f"No se encontró la DB en {persist_directory}. Ejecutá la ingesta primero."
            )

        self.vector_db = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embedding_model
        )

        # 4. Retriever
        self.retriever = self.vector_db.as_retriever(search_kwargs={"k": 1})     #Modificar este valor en el futuro

        # 5. Pipeline RAG
        self.rag_chain = self._build_rag_pipeline()

    # --------------------------------------------------
    # UTILIDADES
    # --------------------------------------------------

    @staticmethod
    def _format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # --------------------------------------------------
    # PIPELINE RAG
    # --------------------------------------------------

    def _build_rag_pipeline(self):
        

        qa_prompt = ChatPromptTemplate.from_messages([
            ("system",
             "Sos un asistente experto en analisis de documentos. Responde usando solo este contexto:\n{context}. Si no sabes"
             " la respuesta, decilo claramente sin inventar nada."
             ),
            ("human", "{input}")
        ])

        rag_chain = (
            {
                "context": itemgetter("input") | self.retriever | RunnableLambda(self._format_docs),
                "input": itemgetter("input"),
            }
            | qa_prompt
            | self.llm
            | StrOutputParser()
        )

        return rag_chain

    # --------------------------------------------------
    # API PÚBLICA
    # --------------------------------------------------

    def chat(self, query: str, chat_history: List[Tuple[str, str]] = []) -> Dict[str, Any]:
        # Convertir historial a mensajes LangChain
        lc_history = []
        for human, ai in chat_history:
            lc_history.append(HumanMessage(content=human))
            lc_history.append(AIMessage(content=ai))

        answer = self.rag_chain.invoke({
            "input": query,
            "chat_history": lc_history
        })

        return {
            "answer": answer,
            "sources": []  # opcional: se puede extender para devolver docs
        }
