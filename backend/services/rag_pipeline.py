from typing import List, Dict, Any
from langchain.schema import Document
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from services.vector_store import VectorStoreService
from config import settings
import logging

logger = logging.getLogger(__name__)


class RAGPipeline:
    def __init__(self):
        # Initialize vector store
        self.vector_store = VectorStoreService()

        # Initialize OpenAI chat model
        self.llm = ChatOpenAI(
            openai_api_key=settings.openai_api_key,
            temperature=0.3,
            model_name="gpt-3.5-turbo"
        )

        # Define prompt template
        self.prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template=(
                "Use the following context to answer the question.\n"
                "If you don't know the answer, just say you don't know.\n\n"
                "Context:\n{context}\n\n"
                "Question:\n{question}\n\n"
                "Answer:"
            )
        )

    def generate_answer(self, question: str, chat_history: List[Dict[str, str]] = None) -> Dict[str, Any]:
        """Generate answer using RAG pipeline"""
        try:
            # Step 1: Retrieve relevant documents
            docs = self._retrieve_documents(question)

            # Step 2: Generate context
            context = self._generate_context(docs)

            # Step 3: Generate LLM answer
            answer = self._generate_llm_response(question, context, chat_history)

            # Step 4: Return structured response
            return {
                "answer": answer,
                "sources": [doc.metadata.get("source", "") for doc in docs],
                "document_count": len(docs)
            }
        except Exception as e:
            logger.error("RAG pipeline failed: %s", str(e))
            return {
                "answer": "An error occurred during answer generation.",
                "sources": [],
                "document_count": 0
            }

    def _retrieve_documents(self, query: str) -> List[Document]:
        """Retrieve relevant documents for the query"""
        top_k = 5  # Can be made configurable
        similarity_threshold = 0.7  # Optional thresholding
        results = self.vector_store.similarity_search(query, k=top_k)

        # Filter by threshold if desired (assumes score is cosine similarity)
        filtered_docs = [doc for doc, score in results if score >= similarity_threshold]
        return filtered_docs

    def _generate_context(self, documents: List[Document]) -> str:
        """Generate context string from documents"""
        if not documents:
            return "No relevant information found."
        return "\n\n".join(doc.page_content for doc in documents)

    def _generate_llm_response(self, question: str, context: str, chat_history: List[Dict[str, str]] = None) -> str:
        """Generate response using LLM"""
        prompt = self.prompt_template.format(context=context, question=question)
        return self.llm.predict(prompt)
