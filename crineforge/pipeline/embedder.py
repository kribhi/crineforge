import os
import gc
import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kwargs: x

from ..utils.logger import get_logger

logger = get_logger(__name__)


class Embedder:
    """
    Generates sentence embeddings using sentence-transformers and builds
    a FAISS index for similarity search.

    Uses a singleton pattern for the embedding model to prevent
    redundant loads (matches the Structurer pattern).
    """

    _model = None
    _model_name = None

    @classmethod
    def get_model(cls, model_name: str = "all-MiniLM-L6-v2"):
        """Lazily loads and caches the SentenceTransformer model (singleton)."""
        if cls._model is not None and cls._model_name == model_name:
            return cls._model

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers is not installed.\n"
                "Reason: Required for embedding generation.\n"
                "Suggested fix: Run `pip install sentence-transformers`"
            )

        logger.info(f"[Embedder] Loading embedding model: {model_name}")
        cls._model = SentenceTransformer(model_name)
        cls._model_name = model_name
        logger.info(f"[Embedder] Model '{model_name}' loaded successfully.")
        return cls._model

    @classmethod
    def free_model(cls) -> None:
        """Releases the embedding model from memory."""
        if cls._model is not None:
            logger.info("[Embedder] Freeing embedding model memory...")
            del cls._model
            cls._model = None
            cls._model_name = None
            gc.collect()
            logger.info("[Embedder] Embedding model freed.")

    @staticmethod
    def embed(
        documents: list[dict],
        output_dir: str,
        model_name: str = "all-MiniLM-L6-v2",
        batch_size: int = 64,
        checkpoint_mgr=None,
    ) -> list[dict]:
        """
        Generates embeddings for all documents and builds a FAISS index.

        Args:
            documents: List of document dicts with 'text' or 'instruction'/'response' keys.
            output_dir: Base directory for saving embeddings and FAISS index.
            model_name: sentence-transformers model identifier.
            batch_size: Encoding batch size.
            checkpoint_mgr: Optional CheckpointManager for resumability.

        Returns:
            Documents with 'embedding_id' attached in metadata.
        """
        if not documents:
            logger.warning("[Embedder] No documents to embed.")
            return []

        logger.info(f"[Embedder] Generating embeddings for {len(documents)} documents...")

        # Extract text content from each document
        texts = []
        for doc in documents:
            text = doc.get("text", "") or ""
            if not text:
                text = f"{doc.get('instruction', '')} {doc.get('response', '')}"
            texts.append(text.strip() or "empty")

        # Generate embeddings
        model = Embedder.get_model(model_name)
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
        )

        # Save vectors as .npy
        embeddings_dir = os.path.join(output_dir, "embeddings")
        os.makedirs(embeddings_dir, exist_ok=True)
        vectors_path = os.path.join(embeddings_dir, "vectors.npy")
        np.save(vectors_path, embeddings)
        logger.info(f"[Embedder] Vectors saved to {vectors_path} (shape: {embeddings.shape})")

        # Build and save FAISS index
        Embedder._build_faiss_index(embeddings, output_dir)

        # Attach embedding_id to documents
        enriched = []
        for idx, doc in enumerate(documents):
            doc_copy = dict(doc)
            metadata = doc_copy.get("metadata", {})
            if not isinstance(metadata, dict):
                metadata = {}
            metadata["embedding_id"] = idx
            doc_copy["metadata"] = metadata
            enriched.append(doc_copy)

        # Save checkpoint
        if checkpoint_mgr:
            checkpoint_mgr.save("stage_05_embed", enriched)
            checkpoint_mgr.mark_complete("stage_05_embed")

        logger.info(f"[Embedder] Embedding complete for {len(enriched)} documents.")
        return enriched

    @staticmethod
    def _build_faiss_index(embeddings: np.ndarray, output_dir: str) -> None:
        """Builds a FAISS IndexFlatL2 and saves it to disk."""
        try:
            import faiss
        except ImportError:
            logger.warning(
                "[Embedder] faiss-cpu not installed — skipping FAISS index creation.\n"
                "Suggested fix: Run `pip install faiss-cpu`"
            )
            return

        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings.astype(np.float32))

        faiss_dir = os.path.join(output_dir, "faiss_index")
        os.makedirs(faiss_dir, exist_ok=True)
        index_path = os.path.join(faiss_dir, "index.faiss")
        faiss.write_index(index, index_path)
        logger.info(f"[Embedder] FAISS index saved to {index_path} ({index.ntotal} vectors)")
