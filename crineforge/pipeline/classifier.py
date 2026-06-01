import re

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kwargs: x

from ..utils.logger import get_logger

logger = get_logger(__name__)


class Classifier:
    """
    Classifies documents into predefined categories using keyword-based
    heuristic rules. No external LLM calls are made.

    Categories: Code, Conversation, Research, Documentation, Tutorial,
    Article, Email, Marketing, Spam, Other.
    """

    CATEGORIES = [
        "Code", "Conversation", "Research", "Documentation",
        "Tutorial", "Article", "Email", "Marketing", "Spam", "Other",
    ]

    # Keyword patterns and their associated weights per category
    _PATTERNS = {
        "Code": {
            "keywords": [
                r'\bdef\s+\w+', r'\bclass\s+\w+', r'\bimport\s+', r'\bfrom\s+\w+\s+import\b',
                r'\bfunction\s+\w+', r'\bconst\s+', r'\blet\s+', r'\bvar\s+',
                r'\breturn\b', r'\bif\s*\(', r'\bfor\s*\(', r'\bwhile\s*\(',
                r'[{}\[\];]', r'=>', r'\bpublic\s+', r'\bprivate\s+',
                r'#include\b', r'\bpackage\b', r'\bfunc\s+\w+',
            ],
            "weight": 2.0,
        },
        "Conversation": {
            "keywords": [
                r'\buser:\s', r'\bassistant:\s', r'\bhuman:\s', r'\bai:\s',
                r'\bQ:\s', r'\bA:\s', r'\?$', r'\bsaid\b', r'\basked\b',
                r'\breplied\b', r'\bthanks\b', r'\bthank you\b',
            ],
            "weight": 1.5,
        },
        "Research": {
            "keywords": [
                r'\babstract\b', r'\bintroduction\b', r'\bmethodology\b',
                r'\bconclusion\b', r'\breferences\b', r'\bcitation\b',
                r'\bhypothesis\b', r'\bexperiment\b', r'\bfindings\b',
                r'\bp\s*[<>=]\s*0\.\d+', r'\bstatistically\b', r'\bcorrelation\b',
                r'\bet\s+al\.', r'\bdoi\b',
            ],
            "weight": 1.5,
        },
        "Documentation": {
            "keywords": [
                r'\bapi\b', r'\bparameters?\b', r'\breturns?\b', r'\bexample:\b',
                r'\busage\b', r'\binstallation\b', r'\bconfiguration\b',
                r'\brequirements?\b', r'\bsetup\b', r'\bdeprecated\b',
                r'```', r'\bnote:\b', r'\bwarning:\b',
            ],
            "weight": 1.3,
        },
        "Tutorial": {
            "keywords": [
                r'\bstep\s+\d+', r'\bhow\s+to\b', r'\bguide\b', r'\btutorial\b',
                r'\blearn\b', r'\bfirst,?\s', r'\bnext,?\s', r'\bthen,?\s',
                r'\bfinally,?\s', r'\bexercise\b', r'\bpractice\b',
                r'\blet\'s\b', r'\bwe\s+will\b',
            ],
            "weight": 1.3,
        },
        "Article": {
            "keywords": [
                r'\bpublished\b', r'\bauthor\b', r'\breported\b',
                r'\baccording\s+to\b', r'\bsource\b', r'\bnews\b',
                r'\bopinion\b', r'\beditor\b', r'\bjournalist\b',
                r'\banalysis\b', r'\bperspective\b',
            ],
            "weight": 1.0,
        },
        "Email": {
            "keywords": [
                r'\bsubject:\s', r'\bfrom:\s', r'\bto:\s', r'\bcc:\s',
                r'\bdear\s+', r'\bsincerely\b', r'\bbest\s+regards\b',
                r'\bfwd:\s', r'\bre:\s', r'\battachment\b',
                r'\bregards\b', r'\bsent\s+from\b',
            ],
            "weight": 2.0,
        },
        "Marketing": {
            "keywords": [
                r'\bbuy\s+now\b', r'\bfree\s+trial\b', r'\bdiscount\b',
                r'\blimited\s+time\b', r'\boffer\b', r'\bsubscribe\b',
                r'\bunsubscribe\b', r'\bpromotion\b', r'\bcall\s+to\s+action\b',
                r'\bclick\s+here\b', r'\bsign\s+up\b',
            ],
            "weight": 1.5,
        },
        "Spam": {
            "keywords": [
                r'\bviagra\b', r'\bcasino\b', r'\blottery\b',
                r'\bwinner\b', r'\bcongratulations\b', r'\bclaim\s+your\b',
                r'\b(?:million|billion)\s+dollars?\b', r'\burgent\b',
                r'\bact\s+now\b', r'\b100%\s+free\b', r'\bguaranteed\b',
                r'\bno\s+risk\b', r'\bwork\s+from\s+home\b',
            ],
            "weight": 2.5,
        },
    }

    @staticmethod
    def classify(documents: list[dict], checkpoint_mgr=None) -> list[dict]:
        """
        Classifies each document and attaches a 'classification' field to its metadata.

        Args:
            documents: List of document dicts with 'text' or 'instruction'/'response' keys.
            checkpoint_mgr: Optional CheckpointManager for resumability.

        Returns:
            Documents with 'classification' metadata attached.
        """
        if not documents:
            logger.warning("[Classifier] No documents to classify.")
            return []

        logger.info(f"[Classifier] Classifying {len(documents)} documents...")
        classified = []

        for doc in tqdm(documents, desc="Classifying"):
            category = Classifier._classify_single(doc)
            doc_copy = dict(doc)
            metadata = doc_copy.get("metadata", {})
            if not isinstance(metadata, dict):
                metadata = {}
            metadata["classification"] = category
            doc_copy["metadata"] = metadata
            classified.append(doc_copy)

        # Log distribution
        distribution = {}
        for doc in classified:
            cat = doc.get("metadata", {}).get("classification", "Other")
            distribution[cat] = distribution.get(cat, 0) + 1
        logger.info(f"[Classifier] Distribution: {distribution}")

        # Save checkpoint
        if checkpoint_mgr:
            checkpoint_mgr.save("stage_04_classify", classified)
            checkpoint_mgr.mark_complete("stage_04_classify")

        return classified

    @staticmethod
    def _classify_single(doc: dict) -> str:
        """Classifies a single document using keyword pattern matching."""
        text = doc.get("text", "") or ""
        if not text:
            text = f"{doc.get('instruction', '')} {doc.get('response', '')}"
        text_lower = text.lower()

        scores = {}
        for category, config in Classifier._PATTERNS.items():
            score = 0.0
            for pattern in config["keywords"]:
                matches = re.findall(pattern, text_lower, re.MULTILINE | re.IGNORECASE)
                score += len(matches)
            scores[category] = score * config["weight"]

        if not scores or max(scores.values()) == 0:
            return "Other"

        return max(scores, key=scores.get)
