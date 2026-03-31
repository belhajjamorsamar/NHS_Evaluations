"""
RAG pipeline orchestration - combines ingestion, vectorization, and retrieval.
Provides high-level interface for the complete RAG workflow.
"""

from typing import List, Optional, Dict, Any
try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document
from src.ingestion import DocumentIngestion
from src.vectorstore import VectorStoreManager
from src.generation import RAGGenerator
from src.config import config
from src.logger import logger


class RAGPipeline:
    """Complete RAG pipeline orchestration."""

    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        embedding_model: Optional[str] = None,
        llm_model: Optional[str] = None,
    ):
        """
        Initialize the complete RAG pipeline.

        Args:
            chunk_size: Document chunk size
            chunk_overlap: Overlap between chunks
            embedding_model: Embedding model name
            llm_model: LLM model name
        """
        self.ingestion = DocumentIngestion(chunk_size, chunk_overlap)
        self.vector_store = VectorStoreManager(embedding_model)
        self.generator = RAGGenerator(llm_model)
        self.is_initialized = False

        logger.info("RAGPipeline initialized")

    def initialize(self, data_directory: Optional[str] = None) -> None:
        """
        Initialize pipeline by loading and indexing documents.

        Args:
            data_directory: Path to documents directory

        Raises:
            RuntimeError: If initialization fails
        """
        try:
            logger.info("Starting pipeline initialization...")

            # Load and chunk documents
            documents = self.ingestion.process_documents(data_directory)

            # Create vector store
            self.vector_store.create_vector_store(documents)

            self.is_initialized = True
            logger.info("Pipeline initialization completed successfully")

        except Exception as e:
            logger.error(f"Pipeline initialization failed: {str(e)}")
            raise RuntimeError(f"Failed to initialize RAG pipeline: {str(e)}")

    def load_existing(self) -> None:
        """
        Load existing vector store without reprocessing documents.

        Raises:
            RuntimeError: If vector store doesn't exist
        """
        try:
            logger.info("Loading existing vector store...")
            self.vector_store.load_vector_store()
            self.is_initialized = True
            logger.info("Existing vector store loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load existing vector store: {str(e)}")
            raise RuntimeError(f"Cannot load vector store: {str(e)}")

    def query(self, question: str, k: Optional[int] = None) -> Dict[str, Any]:
        """
        Query the RAG pipeline with a question.

        Args:
            question: User question
            k: Number of documents to retrieve

        Returns:
            Dictionary with answer, sources, and metadata

        Raises:
            RuntimeError: If pipeline not initialized
            ValueError: If question is empty
        """
        logger.debug(f"Query method called - is_initialized: {self.is_initialized}")
        logger.debug(f"Vector store manager: {self.vector_store}, type: {type(self.vector_store)}")
        logger.debug(f"Vector store internal: {self.vector_store.vector_store if self.vector_store else 'None'}")

        if not self.is_initialized:
            raise RuntimeError("Pipeline not initialized. Call initialize() first.")

        if not question or not question.strip():
            raise ValueError("Question cannot be empty")

        try:
            logger.info(f"Processing query: '{question}'")

            # Validate question is in scope (optional check)
            if self._is_question_out_of_scope(question):
                logger.warning("Question detected as out-of-scope")
                return {
                    "answer": (
                        "Votre question semble ne pas être liée aux services ShopVite. "
                        "Je peux vous aider avec des questions sur les livraisons, retours, "
                        "produits, paiements et support. Comment puis-je vous aider?"
                    ),
                    "sources": [],
                    "confidence": "low",
                    "context_used": 0,
                    "out_of_scope": True,
                }

            # Retrieve relevant documents
            k = k or config.K_RETRIEVALS
            retrieved_docs = self.vector_store.search_similar(question, k=k)

            logger.info(f"Retrieved {len(retrieved_docs)} relevant documents")

            # Generate answer using retrieved context
            result = self.generator.generate_with_fallback(
                question, retrieved_docs
            )

            # Add metadata
            result["out_of_scope"] = False
            result["query"] = question

            logger.info(
                f"Query processed - Confidence: {result.get('confidence')}, "
                f"Sources: {len(result.get('sources', []))}"
            )

            return result

        except Exception as e:
            logger.error(f"Query processing failed: {str(e)}")
            raise RuntimeError(f"Failed to process query: {str(e)}")

    def _is_question_out_of_scope(self, question: str) -> bool:
        """
        Simple heuristic to detect completely off-topic questions.

        Args:
            question: User question

        Returns:
            True if question appears to be completely out of scope
        """
        # E-commerce related keywords
        ecommerce_keywords = [
            "produit", "prix", "livraison", "retour", "paiement",
            "commande", "compte", "garantie", "support", "assistance",
            "shipping", "delivery", "order", "return", "payment",
            "shopvite", "électronique", "produits", "acheter"
        ]

        question_lower = question.lower()

        # Check if question contains any e-commerce keywords
        has_ecommerce_keyword = any(
            keyword in question_lower for keyword in ecommerce_keywords
        )

        # If no e-commerce keywords and very short/vague question
        if not has_ecommerce_keyword and len(question_lower) < 10:
            return True

        return False

    def get_health_status(self) -> Dict[str, Any]:
        """
        Get health status of the pipeline.

        Returns:
            Dictionary with component status
        """
        return {
            "pipeline_initialized": self.is_initialized,
            "vector_store_ready": self.vector_store.vector_store is not None,
            "llm_model": self.generator.model_name,
            "embedding_model": self.vector_store.embedding_model,
        }
