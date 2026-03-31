"""
Document ingestion module for loading and chunking documents.
Supports PDF, TXT, and JSON formats with semantic chunking strategy.
"""

import os
import json
from pathlib import Path
from typing import List, Tuple
try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document
try:
    from langchain_community.document_loaders import (
        PyPDFLoader,
        TextLoader,
        DirectoryLoader,
    )
except ImportError:
    from langchain.document_loaders import (
        PyPDFLoader,
        TextLoader,
        DirectoryLoader,
    )
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
from src.config import config
from src.logger import logger


class DocumentIngestion:
    """Handles document loading and chunking with semantic strategy."""

    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        """
        Initialize document ingestion.

        Args:
            chunk_size: Size of text chunks (default from config)
            chunk_overlap: Overlap between chunks for context preservation
        """
        self.chunk_size = chunk_size or config.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or config.CHUNK_OVERLAP
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        logger.info(
            f"DocumentIngestion initialized: chunk_size={self.chunk_size}, "
            f"overlap={self.chunk_overlap}"
        )

    def load_documents(self, directory: str = None) -> List[Document]:
        """
        Load documents from directory supporting PDF and TXT formats.

        Args:
            directory: Path to documents directory (default from config)

        Returns:
            List of loaded documents

        Raises:
            ValueError: If directory doesn't exist or is empty
        """
        data_dir = directory or config.DATA_DIR
        if not os.path.exists(data_dir):
            raise ValueError(f"Data directory not found: {data_dir}")

        documents = []

        try:
            # Load PDF files
            pdf_loader = DirectoryLoader(
                data_dir, glob="**/*.pdf", loader_cls=PyPDFLoader
            )
            pdf_docs = pdf_loader.load()
            documents.extend(pdf_docs)
            logger.info(f"Loaded {len(pdf_docs)} PDF documents")

            # Load TXT files
            txt_loader = DirectoryLoader(
                data_dir, glob="**/*.txt", loader_cls=TextLoader
            )
            txt_docs = txt_loader.load()
            documents.extend(txt_docs)
            logger.info(f"Loaded {len(txt_docs)} TXT documents")

            # Load JSON files
            json_docs = self._load_json_files(data_dir)
            documents.extend(json_docs)
            logger.info(f"Loaded {len(json_docs)} JSON documents")

            if not documents:
                raise ValueError(f"No documents found in directory: {data_dir}")

            logger.info(f"Total documents loaded: {len(documents)}")
            return documents

        except Exception as e:
            logger.error(f"Error loading documents: {str(e)}")
            raise

    def _load_json_files(self, directory: str) -> List[Document]:
        """
        Load JSON documents from directory.

        Args:
            directory: Path to search for JSON files

        Returns:
            List of Document objects from JSON files
        """
        json_docs = []
        for json_file in Path(directory).rglob("*.json"):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    content = json.dumps(data, ensure_ascii=False, indent=2)
                    doc = Document(
                        page_content=content,
                        metadata={"source": str(json_file), "type": "json"},
                    )
                    json_docs.append(doc)
                    logger.debug(f"Loaded JSON file: {json_file}")
            except Exception as e:
                logger.warning(f"Failed to load JSON file {json_file}: {str(e)}")

        return json_docs

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into semantic chunks.

        Args:
            documents: List of documents to chunk

        Returns:
            List of chunked documents with metadata

        Raises:
            ValueError: If documents list is empty
        """
        if not documents:
            raise ValueError("Documents list cannot be empty")

        try:
            chunked_docs = self.text_splitter.split_documents(documents)
            logger.info(
                f"Documents chunked successfully: {len(documents)} → {len(chunked_docs)} chunks"
            )

            # Add chunk metadata
            for i, doc in enumerate(chunked_docs):
                doc.metadata["chunk_id"] = i
                doc.metadata["chunk_size"] = len(doc.page_content)

            return chunked_docs

        except Exception as e:
            logger.error(f"Error chunking documents: {str(e)}")
            raise

    def process_documents(self, directory: str = None) -> List[Document]:
        """
        Complete pipeline: load and chunk documents.

        Args:
            directory: Path to documents directory (default from config)

        Returns:
            List of processed and chunked documents
        """
        documents = self.load_documents(directory)
        chunked_documents = self.chunk_documents(documents)
        return chunked_documents
