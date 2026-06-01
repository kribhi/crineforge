import re
import math

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kwargs: x

from ..utils.logger import get_logger

logger = get_logger(__name__)


class QualityScorer:
    """
    Scores each document sample from 0.0 to 1.0 on multiple quality dimensions
    using purely heuristic rules (no LLM calls).

    Dimensions scored: coherence, completeness, educational_value,
    information_density, noise_level, spam_likelihood, formatting_quality.

    Documents below the configurable threshold are filtered out with a
    rejection reason attached.
    """

    # Transition words that indicate coherent writing
    _TRANSITION_WORDS = {
        "however", "therefore", "furthermore", "moreover", "additionally",
        "consequently", "nevertheless", "meanwhile", "specifically",
        "for example", "in contrast", "on the other hand", "as a result",
        "in addition", "in conclusion", "first", "second", "third",
        "finally", "next", "then", "also", "thus", "hence",
    }

    @staticmethod
    def score(
        documents: list[dict],
        threshold: float = 0.4,
        checkpoint_mgr=None,
    ) -> list[dict]:
        """
        Scores each document and filters those below the quality threshold.

        Args:
            documents: List of document dicts.
            threshold: Minimum quality score to keep (0.0–1.0).
            checkpoint_mgr: Optional CheckpointManager for resumability.

        Returns:
            Filtered list of documents with quality_score and sub-scores in metadata.
        """
        if not documents:
            logger.warning("[Quality] No documents to score.")
            return []

        logger.info(f"[Quality] Scoring {len(documents)} documents (threshold={threshold})...")

        accepted = []
        rejected_count = 0

        for doc in tqdm(documents, desc="Quality scoring"):
            overall_score, sub_scores, rejection_reason = QualityScorer._score_single(doc, threshold)

            doc_copy = dict(doc)
            metadata = doc_copy.get("metadata", {})
            if not isinstance(metadata, dict):
                metadata = {}

            metadata["quality_score"] = round(overall_score, 4)
            metadata["quality_sub_scores"] = {k: round(v, 4) for k, v in sub_scores.items()}

            if rejection_reason:
                metadata["rejection_reason"] = rejection_reason
                rejected_count += 1
            else:
                doc_copy["metadata"] = metadata
                accepted.append(doc_copy)

        logger.info(
            f"[Quality] Scoring complete: {len(accepted)} accepted, "
            f"{rejected_count} rejected (below {threshold})"
        )

        # Save checkpoint
        if checkpoint_mgr:
            checkpoint_mgr.save("stage_06_quality", accepted)
            checkpoint_mgr.mark_complete("stage_06_quality")

        return accepted

    @staticmethod
    def _score_single(doc: dict, threshold: float) -> tuple[float, dict, str | None]:
        """
        Scores a single document across all quality dimensions.

        Returns:
            (overall_score, sub_scores_dict, rejection_reason_or_None)
        """
        text = doc.get("text", "") or ""
        if not text:
            text = f"{doc.get('instruction', '')} {doc.get('response', '')}"

        text = text.strip()
        if not text:
            return 0.0, {}, "Empty document"

        sub_scores = {
            "coherence": QualityScorer._score_coherence(text),
            "completeness": QualityScorer._score_completeness(text),
            "educational_value": QualityScorer._score_educational_value(text),
            "information_density": QualityScorer._score_information_density(text),
            "noise_level": QualityScorer._score_noise_level(text),
            "spam_likelihood": QualityScorer._score_spam_likelihood(text),
            "formatting_quality": QualityScorer._score_formatting_quality(text),
        }

        overall_score = sum(sub_scores.values()) / len(sub_scores)

        rejection_reason = None
        if overall_score < threshold:
            # Identify the weakest dimension
            weakest = min(sub_scores, key=sub_scores.get)
            rejection_reason = (
                f"Quality score {overall_score:.3f} below threshold {threshold}. "
                f"Weakest dimension: {weakest} ({sub_scores[weakest]:.3f})"
            )

        return overall_score, sub_scores, rejection_reason

    @staticmethod
    def _score_coherence(text: str) -> float:
        """Scores coherence based on sentence structure and transition words."""
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if len(sentences) == 0:
            return 0.0

        # Average sentence length score (ideal: 10-30 words)
        avg_words = sum(len(s.split()) for s in sentences) / len(sentences)
        if 10 <= avg_words <= 30:
            length_score = 1.0
        elif 5 <= avg_words < 10 or 30 < avg_words <= 50:
            length_score = 0.6
        else:
            length_score = 0.3

        # Transition word ratio
        text_lower = text.lower()
        transition_count = sum(1 for tw in QualityScorer._TRANSITION_WORDS if tw in text_lower)
        transition_score = min(transition_count / max(len(sentences), 1), 1.0)

        # Sentence count score (more sentences = more coherent writing)
        count_score = min(len(sentences) / 5.0, 1.0)

        return (length_score * 0.4 + transition_score * 0.3 + count_score * 0.3)

    @staticmethod
    def _score_completeness(text: str) -> float:
        """Scores completeness based on text length brackets."""
        word_count = len(text.split())
        if word_count >= 200:
            return 1.0
        elif word_count >= 100:
            return 0.8
        elif word_count >= 50:
            return 0.6
        elif word_count >= 20:
            return 0.4
        elif word_count >= 5:
            return 0.2
        return 0.0

    @staticmethod
    def _score_educational_value(text: str) -> float:
        """Scores educational value based on explanatory patterns."""
        edu_patterns = [
            r'\bexplain', r'\bbecause\b', r'\btherefore\b', r'\bfor\s+example\b',
            r'\bsuch\s+as\b', r'\bin\s+other\s+words\b', r'\bthis\s+means\b',
            r'\bnote\s+that\b', r'\bimportant', r'\bkey\s+(concept|idea|point)',
            r'\bdefin(e|ition)', r'\bprinciple\b', r'\bconcept\b',
        ]
        text_lower = text.lower()
        matches = sum(1 for p in edu_patterns if re.search(p, text_lower))
        return min(matches / 5.0, 1.0)

    @staticmethod
    def _score_information_density(text: str) -> float:
        """Scores information density via unique word ratio."""
        words = text.lower().split()
        if not words:
            return 0.0
        unique_ratio = len(set(words)) / len(words)
        # Ideal range: 0.4–0.8 unique ratio
        if 0.4 <= unique_ratio <= 0.8:
            return 1.0
        elif 0.3 <= unique_ratio < 0.4 or 0.8 < unique_ratio <= 0.95:
            return 0.7
        return 0.4

    @staticmethod
    def _score_noise_level(text: str) -> float:
        """Scores noise level (inverted — lower noise = higher score)."""
        if not text:
            return 0.0
        # Special character ratio (excluding common punctuation)
        special_chars = sum(1 for c in text if not c.isalnum() and c not in ' \t\n.,!?;:\'"()-')
        special_ratio = special_chars / len(text)
        # Invert: low noise = high score
        return max(0.0, 1.0 - (special_ratio * 5.0))

    @staticmethod
    def _score_spam_likelihood(text: str) -> float:
        """Scores spam likelihood (inverted — low spam = high score)."""
        spam_indicators = [
            r'\bfree\b', r'\bwinner\b', r'\bcongratulations\b', r'\bclick\s+here\b',
            r'\bbuy\s+now\b', r'\blimited\s+time\b', r'\bact\s+now\b',
            r'\b100%\b', r'\bguaranteed\b', r'\b(?:million|billion)\s+dollars?\b',
            r'!!+', r'\$\$+', r'URGENT', r'FREE',
        ]
        text_check = text  # case-sensitive for URGENT, FREE
        matches = sum(1 for p in spam_indicators if re.search(p, text_check, re.IGNORECASE))

        # Check repetition
        words = text.lower().split()
        if len(words) > 5:
            # Count how many consecutive duplicate words exist
            repeats = sum(1 for i in range(1, len(words)) if words[i] == words[i - 1])
            repeat_ratio = repeats / len(words)
        else:
            repeat_ratio = 0.0

        spam_score = min((matches / 3.0) + (repeat_ratio * 2.0), 1.0)
        return max(0.0, 1.0 - spam_score)

    @staticmethod
    def _score_formatting_quality(text: str) -> float:
        """Scores formatting quality based on punctuation and paragraph structure."""
        if not text:
            return 0.0

        # Punctuation ratio
        punctuation = sum(1 for c in text if c in '.!?,;:')
        words = text.split()
        if not words:
            return 0.0
        punct_ratio = punctuation / len(words)

        # Good punctuation ratio: 0.05–0.3
        if 0.05 <= punct_ratio <= 0.3:
            punct_score = 1.0
        elif 0.01 <= punct_ratio < 0.05 or 0.3 < punct_ratio <= 0.5:
            punct_score = 0.6
        else:
            punct_score = 0.3

        # Paragraph structure (presence of line breaks)
        paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
        para_score = min(len(paragraphs) / 3.0, 1.0)

        return punct_score * 0.6 + para_score * 0.4
