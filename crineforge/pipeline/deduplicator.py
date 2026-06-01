import os
import json
import hashlib

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kwargs: x

from ..utils.logger import get_logger

logger = get_logger(__name__)


class Deduplicator:
    """
    Removes exact and near-duplicate documents from the pipeline.

    Supports three deduplication strategies:
    - Exact: SHA256 content hashing
    - MinHash: Locality-sensitive hashing via datasketch for near-duplicates
    - SimHash: Fingerprint-based near-duplicate detection via simhash
    """

    @staticmethod
    def deduplicate(
        documents: list[dict],
        output_dir: str,
        method: str = "all",
        minhash_threshold: float = 0.8,
        checkpoint_mgr=None,
    ) -> tuple[list[dict], dict]:
        """
        Deduplicates documents using the specified method(s).

        Args:
            documents: List of document dicts, each must have a 'text' key.
            output_dir: Directory to write duplicate_report.json.
            method: One of 'exact', 'minhash', 'simhash', or 'all'.
            minhash_threshold: Jaccard similarity threshold for MinHash (0.0–1.0).
            checkpoint_mgr: Optional CheckpointManager for resumability.

        Returns:
            Tuple of (deduplicated_documents, duplicate_report).
        """
        if not documents:
            logger.warning("[Dedup] No documents to deduplicate.")
            return [], {"total_input": 0, "total_output": 0, "duplicates_removed": 0}

        logger.info(f"[Dedup] Starting deduplication on {len(documents)} documents (method={method})")

        all_removed = []
        deduped = list(documents)

        if method in ("exact", "all"):
            deduped, exact_removed = Deduplicator._exact_dedup(deduped)
            all_removed.extend(exact_removed)
            logger.info(f"[Dedup] Exact dedup removed {len(exact_removed)} duplicates")

        if method in ("minhash", "all"):
            deduped, minhash_removed = Deduplicator._minhash_dedup(deduped, minhash_threshold)
            all_removed.extend(minhash_removed)
            logger.info(f"[Dedup] MinHash dedup removed {len(minhash_removed)} near-duplicates")

        if method in ("simhash", "all"):
            deduped, simhash_removed = Deduplicator._simhash_dedup(deduped)
            all_removed.extend(simhash_removed)
            logger.info(f"[Dedup] SimHash dedup removed {len(simhash_removed)} near-duplicates")

        report = {
            "total_input": len(documents),
            "total_output": len(deduped),
            "duplicates_removed": len(all_removed),
            "method": method,
            "minhash_threshold": minhash_threshold,
            "exact_removed": sum(1 for r in all_removed if r.get("reason") == "exact"),
            "minhash_removed": sum(1 for r in all_removed if r.get("reason") == "minhash"),
            "simhash_removed": sum(1 for r in all_removed if r.get("reason") == "simhash"),
        }

        # Write duplicate report
        os.makedirs(output_dir, exist_ok=True)
        report_path = os.path.join(output_dir, "duplicate_report.json")
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        logger.info(f"[Dedup] Duplicate report saved to {report_path}")

        # Save checkpoint
        if checkpoint_mgr:
            checkpoint_mgr.save("stage_03_dedup", deduped)
            checkpoint_mgr.mark_complete("stage_03_dedup")

        return deduped, report

    @staticmethod
    def _exact_dedup(documents: list[dict]) -> tuple[list[dict], list[dict]]:
        """Removes exact duplicates via SHA256 hashing of text content."""
        seen_hashes = set()
        unique = []
        removed = []

        for doc in tqdm(documents, desc="Exact dedup"):
            text = doc.get("text", "") or doc.get("response", "") or ""
            content_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()

            if content_hash in seen_hashes:
                removed.append({"reason": "exact", "hash": content_hash})
            else:
                seen_hashes.add(content_hash)
                unique.append(doc)

        return unique, removed

    @staticmethod
    def _minhash_dedup(
        documents: list[dict], threshold: float = 0.8
    ) -> tuple[list[dict], list[dict]]:
        """Removes near-duplicates using MinHash + LSH from datasketch."""
        try:
            from datasketch import MinHash, MinHashLSH
        except ImportError:
            logger.warning(
                "[Dedup] datasketch not installed — skipping MinHash dedup.\n"
                "Suggested fix: Run `pip install datasketch`"
            )
            return documents, []

        lsh = MinHashLSH(threshold=threshold, num_perm=128)
        minhashes = {}
        unique = []
        removed = []

        for idx, doc in enumerate(tqdm(documents, desc="MinHash dedup")):
            text = doc.get("text", "") or doc.get("response", "") or ""
            words = text.lower().split()

            mh = MinHash(num_perm=128)
            for word in words:
                mh.update(word.encode('utf-8'))

            key = f"doc_{idx}"
            result = lsh.query(mh)

            if result:
                removed.append({"reason": "minhash", "similar_to": result[0]})
            else:
                try:
                    lsh.insert(key, mh)
                    minhashes[key] = mh
                    unique.append(doc)
                except ValueError:
                    # Duplicate key — treat as duplicate
                    removed.append({"reason": "minhash", "similar_to": key})

        return unique, removed

    @staticmethod
    def _simhash_dedup(documents: list[dict], distance_threshold: int = 3) -> tuple[list[dict], list[dict]]:
        """Removes near-duplicates using SimHash fingerprinting."""
        try:
            from simhash import Simhash
        except ImportError:
            logger.warning(
                "[Dedup] simhash not installed — skipping SimHash dedup.\n"
                "Suggested fix: Run `pip install simhash`"
            )
            return documents, []

        fingerprints = []
        unique = []
        removed = []

        for doc in tqdm(documents, desc="SimHash dedup"):
            text = doc.get("text", "") or doc.get("response", "") or ""
            sh = Simhash(text)

            is_dup = False
            for existing_sh in fingerprints:
                if sh.distance(existing_sh) <= distance_threshold:
                    is_dup = True
                    break

            if is_dup:
                removed.append({"reason": "simhash"})
            else:
                fingerprints.append(sh)
                unique.append(doc)

        return unique, removed
