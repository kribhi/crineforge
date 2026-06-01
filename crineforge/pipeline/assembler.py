import os
import json
import random

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kwargs: x

from ..utils.logger import get_logger

logger = get_logger(__name__)


class Assembler:
    """
    Splits the processed dataset into train/valid/test splits and generates
    a comprehensive statistics report.
    """

    @staticmethod
    def assemble(
        documents: list[dict],
        output_dir: str,
        train_split: float = 0.8,
        valid_split: float = 0.1,
        test_split: float = 0.1,
        seed: int = 42,
        checkpoint_mgr=None,
    ) -> dict:
        """
        Assembles the final dataset by splitting into train/valid/test.

        Args:
            documents: List of processed document dicts.
            output_dir: Base output directory.
            train_split: Fraction for training set (default 0.8).
            valid_split: Fraction for validation set (default 0.1).
            test_split: Fraction for test set (default 0.1).
            seed: Random seed for reproducible splits.
            checkpoint_mgr: Optional CheckpointManager for resumability.

        Returns:
            Dataset statistics report dict.

        Raises:
            ValueError: If splits don't sum to 1.0 or no documents provided.
        """
        # Validate splits
        total = round(train_split + valid_split + test_split, 2)
        if abs(total - 1.0) > 0.01:
            raise ValueError(
                f"Split fractions must sum to 1.0, got {total}.\n"
                f"Reason: train={train_split} + valid={valid_split} + test={test_split} = {total}\n"
                f"Suggested fix: Adjust split values to sum to 1.0."
            )

        if not documents:
            logger.warning("[Assembler] No documents to assemble.")
            return {"total_samples": 0}

        logger.info(
            f"[Assembler] Assembling {len(documents)} documents "
            f"(splits: {train_split}/{valid_split}/{test_split})..."
        )

        # Shuffle deterministically
        shuffled = list(documents)
        random.seed(seed)
        random.shuffle(shuffled)

        # Compute split indices
        n = len(shuffled)
        train_end = int(n * train_split)
        valid_end = train_end + int(n * valid_split)

        train_set = shuffled[:train_end]
        valid_set = shuffled[train_end:valid_end]
        test_set = shuffled[valid_end:]

        logger.info(
            f"[Assembler] Split sizes — train: {len(train_set)}, "
            f"valid: {len(valid_set)}, test: {len(test_set)}"
        )

        # Write JSONL files
        dataset_dir = os.path.join(output_dir, "clean_dataset")
        os.makedirs(dataset_dir, exist_ok=True)

        Assembler._write_jsonl(train_set, os.path.join(dataset_dir, "train.jsonl"))
        Assembler._write_jsonl(valid_set, os.path.join(dataset_dir, "valid.jsonl"))
        Assembler._write_jsonl(test_set, os.path.join(dataset_dir, "test.jsonl"))

        # Generate report
        report = Assembler._generate_report(train_set, valid_set, test_set, documents)

        report_path = os.path.join(output_dir, "dataset_report.json")
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        logger.info(f"[Assembler] Dataset report saved to {report_path}")

        # Save checkpoint
        if checkpoint_mgr:
            checkpoint_mgr.save("stage_08_assemble", documents)
            checkpoint_mgr.mark_complete("stage_08_assemble")

        return report

    @staticmethod
    def _write_jsonl(data: list[dict], path: str) -> None:
        """Writes a list of dicts to a JSONL file."""
        with open(path, 'w', encoding='utf-8') as f:
            for record in data:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        logger.info(f"[Assembler] Wrote {len(data)} records to {path}")

    @staticmethod
    def _generate_report(
        train: list[dict],
        valid: list[dict],
        test: list[dict],
        all_docs: list[dict],
    ) -> dict:
        """Generates comprehensive dataset statistics."""
        report = {
            "total_samples": len(all_docs),
            "splits": {
                "train": len(train),
                "valid": len(valid),
                "test": len(test),
            },
        }

        # Language distribution
        lang_dist = {}
        for doc in all_docs:
            lang = doc.get("metadata", {}).get("language", "unknown")
            lang_dist[lang] = lang_dist.get(lang, 0) + 1
        report["language_distribution"] = lang_dist

        # Category breakdown
        cat_dist = {}
        for doc in all_docs:
            cat = doc.get("metadata", {}).get("classification", "Other")
            cat_dist[cat] = cat_dist.get(cat, 0) + 1
        report["category_breakdown"] = cat_dist

        # Quality score histogram
        scores = []
        for doc in all_docs:
            qs = doc.get("metadata", {}).get("quality_score")
            if qs is not None:
                scores.append(qs)

        if scores:
            bins = {"0.0-0.2": 0, "0.2-0.4": 0, "0.4-0.6": 0, "0.6-0.8": 0, "0.8-1.0": 0}
            for s in scores:
                if s < 0.2:
                    bins["0.0-0.2"] += 1
                elif s < 0.4:
                    bins["0.2-0.4"] += 1
                elif s < 0.6:
                    bins["0.4-0.6"] += 1
                elif s < 0.8:
                    bins["0.6-0.8"] += 1
                else:
                    bins["0.8-1.0"] += 1

            report["quality_score_histogram"] = bins
            report["quality_score_stats"] = {
                "mean": round(sum(scores) / len(scores), 4),
                "min": round(min(scores), 4),
                "max": round(max(scores), 4),
            }

        # Average text length
        lengths = []
        for doc in all_docs:
            text = doc.get("text", "") or doc.get("response", "") or ""
            lengths.append(len(text.split()))
        if lengths:
            report["avg_word_count"] = round(sum(lengths) / len(lengths), 1)

        return report
