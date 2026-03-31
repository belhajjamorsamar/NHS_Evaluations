"""
Augmented generation module for LLM-based answer generation.
Handles LLM calls with context injection and error handling.
"""

from typing import List, Dict, Any
try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document
try:
    from langchain_openai import ChatOpenAI
except ImportError:
    from langchain.chat_models import ChatOpenAI
try:
    from langchain_core.prompts import PromptTemplate
except ImportError:
    from langchain.prompts import PromptTemplate
try:
    from langchain.chains import LLMChain
except ImportError:
    from langchain_core.runnables import RunnablePassthrough
from src.config import config
from src.prompts import (
    get_system_prompt,
    get_retrieval_prompt_template,
)
from src.logger import logger


class RAGGenerator:
    """Handles RAG-based answer generation with LLM."""

    def __init__(self, model_name: str = None):
        """
        Initialize RAG generator.

        Args:
            model_name: OpenAI model name (default from config)
        """
        self.model_name = model_name or config.OPENAI_MODEL
        self.llm = ChatOpenAI(
            model=self.model_name,
            api_key=config.OPENAI_API_KEY,
            base_url=config.OPENAI_BASE_URL,
            temperature=0.7,
            max_tokens=1000,
            request_timeout=30,
        )
        self.system_prompt = get_system_prompt()
        logger.info(f"RAGGenerator initialized with model: {self.model_name}")

    def format_context(self, documents: List[Document]) -> str:
        """
        Format retrieved documents into context string.

        Args:
            documents: List of relevant documents

        Returns:
            Formatted context string with sources
        """
        context_parts = []

        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get("source", "Unknown")
            content = doc.page_content[:500]  # Limit per document

            context_parts.append(
                f"[Document {i}] (Source: {source})\n{content}\n"
            )

        return "\n---\n".join(context_parts)

    def extract_sources(self, documents: List[Document]) -> List[str]:
        """
        Extract unique source citations from documents.

        Args:
            documents: List of documents

        Returns:
            List of source references
        """
        sources = []
        seen = set()

        for doc in documents:
            source = doc.metadata.get("source", "Unknown")
            chunk_id = doc.metadata.get("chunk_id", 0)

            if source not in seen:
                sources.append(source)
                seen.add(source)

        return sources

    def generate_answer(
        self,
        question: str,
        documents: List[Document],
        user_context: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """
        Generate answer based on question and retrieved documents.

        Args:
            question: User question
            documents: Retrieved relevant documents
            user_context: Optional additional context

        Returns:
            Dictionary with answer, sources, and confidence

        Raises:
            RuntimeError: If LLM call fails
        """
        if not documents:
            return {
                "answer": (
                    "Je n'ai pas trouvé d'informations pertinentes "
                    "pour répondre à votre question."
                ),
                "sources": [],
                "confidence": "low",
                "context_used": 0,
            }

        try:
            # Format context from documents
            context = self.format_context(documents)

            # Create the retrieval prompt
            prompt_template = PromptTemplate(
                input_variables=["context", "question"],
                template=get_retrieval_prompt_template(),
            )

            logger.debug(
                f"Generating answer for question: '{question}' with {len(documents)} documents"
            )

            # Generate answer using modern LangChain approach
            # Format the prompt with context and question
            formatted_prompt = prompt_template.format(context=context, question=question)

            # Call the LLM directly
            response = self.llm.invoke(formatted_prompt)
            # Extract text from the response
            if hasattr(response, 'content'):
                response = response.content
            else:
                response = str(response)

            # Extract sources
            sources = self.extract_sources(documents)

            # Determine confidence level based on number of sources
            if len(documents) >= 3:
                confidence = "high"
            elif len(documents) >= 1:
                confidence = "medium"
            else:
                confidence = "low"

            result = {
                "answer": response.strip(),
                "sources": sources,
                "confidence": confidence,
                "context_used": len(documents),
                "model": self.model_name,
            }

            logger.info(f"Answer generated successfully with confidence: {confidence}")
            return result

        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            raise RuntimeError(f"Failed to generate answer: {str(e)}")

    def generate_with_fallback(
        self,
        question: str,
        documents: List[Document],
        max_retries: int = 2,
    ) -> Dict[str, Any]:
        """
        Generate answer with fallback mechanism for failures.

        Args:
            question: User question
            documents: Retrieved documents
            max_retries: Maximum retry attempts

        Returns:
            Answer dictionary or fallback response
        """
        for attempt in range(max_retries):
            try:
                return self.generate_answer(question, documents)
            except Exception as e:
                logger.warning(
                    f"Generation attempt {attempt + 1}/{max_retries} failed: {str(e)}"
                )

                if attempt == max_retries - 1:
                    # Return fallback response
                    logger.error("All generation attempts failed, returning fallback")
                    return {
                        "answer": (
                            "Je suis temporairement indisponible. "
                            "Veuillez réessayer dans quelques instants."
                        ),
                        "sources": [],
                        "confidence": "low",
                        "context_used": 0,
                        "model": self.model_name,
                    }

        return {}
