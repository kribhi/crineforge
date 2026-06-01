import os
import re

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kwargs: x

from ..utils.logger import get_logger

logger = get_logger(__name__)


class Enricher:
    """
    Generates metadata for each document (tags, categories, language,
    source_type, quality_score) and converts documents to the target
    training format (instruction, chat, or completion).
    """

    @staticmethod
    def enrich(
        documents: list[dict],
        dataset_format: str = "instruction",
        checkpoint_mgr=None,
    ) -> list[dict]:
        """
        Enriches documents with metadata and converts to target format.

        Args:
            documents: List of document dicts.
            dataset_format: One of 'instruction', 'chat', 'completion'.
            checkpoint_mgr: Optional CheckpointManager for resumability.

        Returns:
            Enriched and format-converted documents.
        """
        if not documents:
            logger.warning("[Enricher] No documents to enrich.")
            return []

        logger.info(
            f"[Enricher] Enriching {len(documents)} documents "
            f"(format={dataset_format})..."
        )

        enriched = []
        for doc in tqdm(documents, desc="Enriching"):
            doc_copy = dict(doc)
            metadata = doc_copy.get("metadata", {})
            if not isinstance(metadata, dict):
                metadata = {}

            # Extract text for analysis
            text = doc_copy.get("text", "") or ""
            if not text:
                text = f"{doc_copy.get('instruction', '')} {doc_copy.get('response', '')}"

            # Generate metadata
            metadata["language"] = Enricher._detect_language(text)
            metadata["tags"] = Enricher._generate_tags(doc_copy)
            metadata["source_type"] = metadata.get("source_type", "unknown")

            doc_copy["metadata"] = metadata

            # Convert to target format
            if dataset_format == "instruction":
                doc_copy = Enricher._to_instruction_format(doc_copy)
            elif dataset_format == "chat":
                doc_copy = Enricher._to_chat_format(doc_copy)
            elif dataset_format == "completion":
                doc_copy = Enricher._to_completion_format(doc_copy)

            enriched.append(doc_copy)

        # Log language distribution
        lang_dist = {}
        for doc in enriched:
            lang = doc.get("metadata", {}).get("language", "unknown")
            lang_dist[lang] = lang_dist.get(lang, 0) + 1
        logger.info(f"[Enricher] Language distribution: {lang_dist}")

        # Save checkpoint
        if checkpoint_mgr:
            checkpoint_mgr.save("stage_07_enrich", enriched)
            checkpoint_mgr.mark_complete("stage_07_enrich")

        return enriched

    @staticmethod
    def _detect_language(text: str) -> str:
        """Detects the language of the text using langdetect."""
        if not text or len(text.strip()) < 20:
            return "unknown"

        try:
            from langdetect import detect
            return detect(text[:5000])  # Limit text length for performance
        except ImportError:
            logger.warning(
                "[Enricher] langdetect not installed — defaulting to 'en'.\n"
                "Suggested fix: Run `pip install langdetect`"
            )
            return "en"
        except Exception:
            return "unknown"

    @staticmethod
    def _generate_tags(doc: dict) -> list[str]:
        """Generates tags via simple keyword frequency extraction."""
        text = doc.get("text", "") or ""
        if not text:
            text = f"{doc.get('instruction', '')} {doc.get('response', '')}"

        # Tokenize and count
        words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
        if not words:
            return []

        # Stopwords (minimal set for speed)
        stopwords = {
            "that", "this", "with", "from", "have", "been", "were", "they",
            "their", "what", "when", "where", "which", "will", "would",
            "could", "should", "about", "there", "these", "those", "than",
            "then", "into", "some", "such", "each", "also", "more",
            "other", "very", "just", "only", "your", "does",
        }

        freq = {}
        for word in words:
            if word not in stopwords:
                freq[word] = freq.get(word, 0) + 1

        # Top 5 most frequent words as tags
        sorted_words = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, count in sorted_words[:5]]

    @staticmethod
    def _to_instruction_format(doc: dict) -> dict:
        """Converts document to instruction-tuning format."""
        result = dict(doc)
        if "instruction" not in result and "text" in result:
            text = result["text"]
            # Split at first natural break or use as-is
            sentences = re.split(r'(?<=[.!?])\s+', text, maxsplit=1)
            if len(sentences) >= 2:
                result["instruction"] = sentences[0]
                result["response"] = sentences[1]
            else:
                result["instruction"] = "Summarize or explain the following:"
                result["response"] = text
        return result

    @staticmethod
    def _to_chat_format(doc: dict) -> dict:
        """Converts document to chat format with user/assistant roles."""
        result = dict(doc)
        instruction = result.get("instruction", "")
        response = result.get("response", "")

        if not instruction and "text" in result:
            text = result["text"]
            sentences = re.split(r'(?<=[.!?])\s+', text, maxsplit=1)
            if len(sentences) >= 2:
                instruction = sentences[0]
                response = sentences[1]
            else:
                instruction = text
                response = ""

        result["messages"] = [
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": response},
        ]
        return result

    @staticmethod
    def _to_completion_format(doc: dict) -> dict:
        """Converts document to completion format (single text field)."""
        result = dict(doc)
        instruction = result.get("instruction", "")
        response = result.get("response", "")

        if instruction and response:
            result["text"] = f"{instruction}\n\n{response}"
        elif "text" not in result:
            result["text"] = f"{instruction} {response}".strip()

        return result
