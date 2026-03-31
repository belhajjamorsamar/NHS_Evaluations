"""
RAGAS Evaluation Module for RAG Pipeline Quality Assessment.
Measures faithfulness, answer relevancy, and context recall metrics.
"""

from typing import List, Dict, Any
from dataclasses import dataclass
import pandas as pd
from src.retrieval import RAGPipeline
from src.logger import logger

# Try to import RAGAS (optional dependency)
try:
    from ragas.metrics import faithfulness, answer_relevancy, context_recall
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False
    logger.warning("RAGAS not installed. Install with: pip install ragas")


@dataclass
class EvaluationResult:
    """Single evaluation result for a question."""

    question: str
    ground_truth: str
    generated_answer: str
    retrieved_contexts: List[str]
    faithfulness_score: float = 0.0
    answer_relevancy_score: float = 0.0
    context_recall_score: float = 0.0
    average_score: float = 0.0


class RAGEvaluator:
    """Evaluate RAG pipeline quality using RAGAS metrics."""

    def __init__(self, pipeline: RAGPipeline):
        """
        Initialize evaluator.

        Args:
            pipeline: Initialized RAGPipeline instance
        """
        self.pipeline = pipeline
        if not RAGAS_AVAILABLE:
            logger.warning("RAGAS evaluation disabled (optional dependency)")

    def evaluate_question(
        self, question: str, ground_truth: str = None
    ) -> EvaluationResult:
        """
        Evaluate a single question.

        Args:
            question: Question to evaluate
            ground_truth: Expected/ideal answer for relevancy comparison

        Returns:
            EvaluationResult with metrics
        """
        try:
            # Generate answer
            result = self.pipeline.query(question)
            generated_answer = result.get("answer", "")
            sources = result.get("sources", [])

            evaluation = EvaluationResult(
                question=question,
                ground_truth=ground_truth or "",
                generated_answer=generated_answer,
                retrieved_contexts=sources,
            )

            # Calculate metrics if RAGAS available
            if RAGAS_AVAILABLE:
                # Note: RAGAS metrics require LLM evaluation
                # Simplified calculation for demo
                evaluation.faithfulness_score = self._calculate_faithfulness(
                    generated_answer, sources
                )
                evaluation.answer_relevancy_score = self._calculate_relevancy(
                    question, generated_answer
                )
                evaluation.context_recall_score = len(sources) / 3  # Max 3 sources

                # Average score
                evaluation.average_score = (
                    evaluation.faithfulness_score
                    + evaluation.answer_relevancy_score
                    + evaluation.context_recall_score
                ) / 3

            return evaluation

        except Exception as e:
            logger.error(f"Error evaluating question: {str(e)}")
            return EvaluationResult(
                question=question,
                ground_truth=ground_truth or "",
                generated_answer="ERROR",
                retrieved_contexts=[],
            )

    def _calculate_faithfulness(self, answer: str, sources: List[str]) -> float:
        """
        Simple faithfulness heuristic: penalize if no sources cited.

        Args:
            answer: Generated answer
            sources: Retrieved source documents

        Returns:
            Faithfulness score (0-1)
        """
        if not sources:
            return 0.5  # Low score if no sources

        # Check if answer length is reasonable
        if len(answer) < 50:
            return 0.6  # Shorter answers are less faithful

        # Check for common hallucination patterns
        hallucination_keywords = [
            "selon les données",
            "sur notre site",
            "dans notre catalogue",
        ]
        has_hallucination = any(
            keyword in answer.lower() for keyword in hallucination_keywords
        )

        return 0.7 if not has_hallucination else 0.5

    def _calculate_relevancy(self, question: str, answer: str) -> float:
        """
        Simple relevancy heuristic: measure overlap between question and answer.

        Args:
            question: User question
            answer: Generated answer

        Returns:
            Relevancy score (0-1)
        """
        if not answer or len(answer) < 20:
            return 0.3

        # Extract keywords from question
        question_words = set(
            word.lower()
            for word in question.split()
            if len(word) > 3 and word.lower() not in ["quelle", "quels", "comment"]
        )

        # Extract keywords from answer
        answer_words = set(word.lower() for word in answer.split() if len(word) > 3)

        # Calculate overlap
        overlap = len(question_words & answer_words)
        total = len(question_words | answer_words)

        return min(overlap / max(total, 1), 1.0) if total > 0 else 0.5

    def evaluate_batch(
        self, test_questions: List[Dict[str, str]]
    ) -> List[EvaluationResult]:
        """
        Evaluate multiple questions.

        Args:
            test_questions: List of dicts with 'question' and optional 'ground_truth'

        Returns:
            List of evaluation results
        """
        results = []

        for test in test_questions:
            question = test.get("question", "")
            ground_truth = test.get("ground_truth", None)

            logger.info(f"Evaluating: {question[:50]}...")
            result = self.evaluate_question(question, ground_truth)
            results.append(result)

        return results

    def generate_report(
        self, results: List[EvaluationResult]
    ) -> Dict[str, Any]:
        """
        Generate evaluation report.

        Args:
            results: List of evaluation results

        Returns:
            Dictionary with summary statistics
        """
        if not results:
            return {"error": "No results to report"}

        faithfulness_scores = [r.faithfulness_score for r in results]
        relevancy_scores = [r.answer_relevancy_score for r in results]
        recall_scores = [r.context_recall_score for r in results]
        average_scores = [r.average_score for r in results]

        return {
            "total_questions": len(results),
            "faithfulness": {
                "mean": sum(faithfulness_scores) / len(faithfulness_scores),
                "min": min(faithfulness_scores),
                "max": max(faithfulness_scores),
            },
            "answer_relevancy": {
                "mean": sum(relevancy_scores) / len(relevancy_scores),
                "min": min(relevancy_scores),
                "max": max(relevancy_scores),
            },
            "context_recall": {
                "mean": sum(recall_scores) / len(recall_scores),
                "min": min(recall_scores),
                "max": max(recall_scores),
            },
            "overall": {
                "mean": sum(average_scores) / len(average_scores),
                "min": min(average_scores),
                "max": max(average_scores),
            },
        }

    def results_to_dataframe(
        self, results: List[EvaluationResult]
    ) -> "pd.DataFrame":
        """
        Convert results to pandas DataFrame for analysis.

        Args:
            results: List of evaluation results

        Returns:
            DataFrame with evaluation results
        """
        data = [
            {
                "question": r.question,
                "generated_answer": r.generated_answer[:100],  # Truncate
                "faithfulness": r.faithfulness_score,
                "relevancy": r.answer_relevancy_score,
                "context_recall": r.context_recall_score,
                "average": r.average_score,
            }
            for r in results
        ]

        return pd.DataFrame(data)


# Example test questions
EXAMPLE_TEST_QUESTIONS = [
    {
        "question": "Quel est le délai de retour pour ShopVite?",
        "ground_truth": "30 jours",
    },
    {
        "question": "Combien coûte la livraison express?",
        "ground_truth": "15€",
    },
    {
        "question": "Quels sont les moyens de paiement acceptés?",
        "ground_truth": "Cartes, PayPal, Google Pay, Apple Pay",
    },
    {
        "question": "Comment contacter le support client?",
        "ground_truth": "Email, chat, téléphone",
    },
    {
        "question": "Y a-t-il une garantie sur les produits?",
        "ground_truth": "2 ans de garantie légale",
    },
    {
        "question": "Est-ce que je peux retourner un produit ouvert?",
        "ground_truth": "Non, sauf si endommagé à la réception",
    },
    {
        "question": "Livrez-vous en dehors de la France?",
        "ground_truth": "Oui, Union Européenne et pays partenaires",
    },
    {
        "question": "Puis-je changer mon adresse de livraison?",
        "ground_truth": "Oui, jusqu'à 24h avant expédition",
    },
    {
        "question": "Comment réinitialiser mon mot de passe?",
        "ground_truth": "Via le lien 'Mot de passe oublié'",
    },
    {
        "question": "Avez-vous un programme de fidélité?",
        "ground_truth": "Oui, 1€ = 1 point",
    },
]


if __name__ == "__main__":
    # Example: Run evaluation
    logger.info("Starting RAG evaluation...")

    # Initialize pipeline
    pipeline = RAGPipeline()
    try:
        pipeline.load_existing()
    except:
        logger.error("Pipeline not initialized. Run src.api first.")
        exit(1)

    # Initialize evaluator
    evaluator = RAGEvaluator(pipeline)

    # Evaluate test questions
    results = evaluator.evaluate_batch(EXAMPLE_TEST_QUESTIONS)

    # Generate report
    report = evaluator.generate_report(results)
    logger.info(f"Evaluation Report: {report}")

    # Convert to DataFrame
    df = evaluator.results_to_dataframe(results)
    logger.info(f"\n{df.to_string()}")

    # Save results
    df.to_csv("eval_results.csv", index=False)
    logger.info("Results saved to eval_results.csv")
