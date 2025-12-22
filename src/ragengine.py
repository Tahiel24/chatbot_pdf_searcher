import os
from typing import List, Tuple, Dict, Any
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter


class RAGEngine:
    def __init__(self, persist_directory: str = "data/vector_store"):
        self.persist_directory = persist_directory

        # 1. Embeddings (idéntico a la ingesta)
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # 2. LLM local (Ollama)
        self.llm = ChatOllama(
            model="llama3.2:1b",     #Cambiar esta linea segun el modelo llm que se vaya a utilizar
            temperature=0,
            keep_alive="5m"
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
        self.retriever = self.vector_db.as_retriever(search_kwargs={"k": 3})

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
        # ---------- ETAPA 1: Reescritura de la pregunta ----------
        contextualize_prompt = ChatPromptTemplate.from_messages([
            ("system",
             "Dado el historial y la pregunta del usuario, reformulá la pregunta "
             "para que sea independiente. NO la respondas."),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])

        contextualize_chain = (
            contextualize_prompt
            | self.llm
            | StrOutputParser()
        )

        # ---------- ETAPA 2: Retrieval ----------
        def retrieve_docs(question: str):
            return self.retriever.invoke(question)

        # ---------- ETAPA 3 + 4: QA ----------
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system",
             "Sos un asistente experto en análisis documental.\n"
             "Usá SOLO el contexto para responder.\n\n"
             "Reglas:\n"
             "1. Si no sabés, decilo.\n"
             "2. Respuesta clara y profesional.\n"
             "3. Citá fuente y página.\n\n"
             "Contexto:\n{context}"
             ),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])

        rag_chain = (
            {
                "standalone_question": contextualize_chain,
                "context": contextualize_chain | retrieve_docs | self._format_docs,
                "input": itemgetter("input"),
                "chat_history": itemgetter("chat_history")
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
