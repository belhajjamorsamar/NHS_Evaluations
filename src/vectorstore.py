"""
Vector store management for embeddings and similarity search.
Uses ChromaDB for persistent local vector storage.
"""

from typing import List, Optional
try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document
try:
    from langchain_openai import OpenAIEmbeddings
except ImportError:
    from langchain.embeddings.openai import OpenAIEmbeddings
try:
    from langchain_community.vectorstores import Chroma
except ImportError:
    from langchain.vectorstores import Chroma
from src.config import config
from src.logger import logger


class VectorStoreManager:
    """Manages vector store operations for document embeddings."""

    def __init__(self, embedding_model: str = None, persist_dir: str = None):
        """
        Initialize vector store manager.

        Args:
            embedding_model: OpenAI embedding model name
            persist_dir: Directory for persistent storage
        """
        self.embedding_model = embedding_model or config.OPENAI_EMBEDDING_MODEL
        self.persist_dir = persist_dir or config.PERSIST_DIRECTORY
        self.embeddings = OpenAIEmbeddings(
            model=self.embedding_model,
            api_key=config.OPENAI_API_KEY,
            base_url=config.OPENAI_BASE_URL,
        )
        self.vector_store: Optional[Chroma] = None
        logger.info(
            f"VectorStoreManager initialized with model: {self.embedding_model}"
        )

    def create_vector_store(self, documents: List[Document]) -> Chroma:
        """
        Create and persist vector store from documents.

        Args:
            documents: List of Document objects to index

        Returns:
            Chroma vector store instance

        Raises:
            ValueError: If documents list is empty
        """
        if not documents:
            raise ValueError("Documents list cannot be empty")

        try:
            logger.info(f"Creating vector store with {len(documents)} documents...")

            self.vector_store = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=self.persist_dir,
            )

            logger.info(
                f"Vector store created successfully at: {self.persist_dir}"
            )
            return self.vector_store

        except Exception as e:
            logger.error(f"Error creating vector store: {str(e)}")
            raise

    def load_vector_store(self) -> Chroma:
        """
        Load existing vector store from persistence.

        Returns:
            Loaded Chroma vector store

        Raises:
            RuntimeError: If vector store doesn't exist
        """
        try:
            logger.info(f"Loading vector store from: {self.persist_dir}")

            self.vector_store = Chroma(
                persist_directory=self.persist_dir,
                embedding_function=self.embeddings,
            )

            logger.info("Vector store loaded successfully")
            return self.vector_store

        except Exception as e:
            logger.error(f"Error loading vector store: {str(e)}")
            raise

    def get_or_create_vector_store(
        self, documents: List[Document] = None
    ) -> Chroma:
        """
        Get existing vector store or create new one if documents provided.

        Args:
            documents: Optional list of documents to create store

        Returns:
            Chroma vector store instance
        """
        try:
            # Try loading existing store
            return self.load_vector_store()
        except Exception as e:
            logger.warning(f"Could not load existing vector store: {str(e)}")

            # Create new store if documents provided
            if documents:
                return self.create_vector_store(documents)
            else:
                raise RuntimeError(
                    "Vector store not found and no documents provided to create new one"
                )

    def search_similar(
        self, query: str, k: int = None, threshold: float = None
    ) -> List[Document]:
        """
        Search for k most similar documents using similarity search.

        Args:
            query: Query text
            k: Number of results (default from config)
            threshold: Minimum similarity score (default from config)

        Returns:
            List of similar documents

        Raises:
            RuntimeError: If vector store not initialized
        """
        logger.debug(f"search_similar called - vector_store type: {type(self.vector_store)}")
        if self.vector_store is None:
            logger.error(f"Vector store is None!")
            raise RuntimeError("Vector store not initialized")

        k = k or config.K_RETRIEVALS
        threshold = threshold or config.SIMILARITY_THRESHOLD

        try:
            logger.debug(f"Searching for similar documents: query='{query}', k={k}")

            # Perform similarity search with score
            results_with_scores = (
                self.vector_store.similarity_search_with_score(query, k=k)
            )

            # Filter by threshold
            results = [
                doc
                for doc, score in results_with_scores
                if score <= (1 - threshold)  # ChromaDB returns distance, not similarity
            ]

            logger.debug(f"Found {len(results)} relevant documents")
            # Return results or fall back to top 1 document if no threshold matches
            if results:
                return results
            elif results_with_scores:
                return [results_with_scores[0][0]]  # Extract just the document from (doc, score) tuple
            else:
                return []

        except Exception as e:
            logger.error(f"Error searching vector store: {str(e)}")
            raise

    def clear_vector_store(self) -> None:
        """Clear all documents from vector store."""
        if self.vector_store:
            try:
                # Delete the collection
                self.vector_store._collection.delete(
                    where={}
                )  # Delete all documents
                logger.info("Vector store cleared")
            except Exception as e:
                logger.warning(f"Error clearing vector store: {str(e)}")
