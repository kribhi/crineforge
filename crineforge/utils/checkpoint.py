import os
import json
from .logger import get_logger

logger = get_logger(__name__)


class CheckpointManager:
    """
    Manages stage-level JSONL checkpoints for pipeline resumability.

    Each pipeline stage writes its intermediate results to a JSONL checkpoint file.
    If the pipeline is interrupted, stages can resume from the last checkpoint
    instead of re-processing from scratch.
    """

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.checkpoint_dir = os.path.join(output_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        logger.info(f"[Checkpoint] Initialized checkpoint directory: {self.checkpoint_dir}")

    def _get_checkpoint_path(self, stage_id: str) -> str:
        """Returns the JSONL checkpoint file path for a given stage."""
        return os.path.join(self.checkpoint_dir, f"{stage_id}.jsonl")

    def _get_flag_path(self, stage_id: str) -> str:
        """Returns the completion flag file path for a given stage."""
        return os.path.join(self.checkpoint_dir, f"{stage_id}_complete")

    def save(self, stage_id: str, data_batch: list[dict]) -> None:
        """
        Appends a batch of records to the stage's JSONL checkpoint file.
        Flushes immediately to ensure crash safety.
        """
        checkpoint_path = self._get_checkpoint_path(stage_id)
        try:
            with open(checkpoint_path, 'a', encoding='utf-8') as f:
                for record in data_batch:
                    f.write(json.dumps(record, ensure_ascii=False) + '\n')
                f.flush()
                os.fsync(f.fileno())
            logger.debug(f"[Checkpoint] Saved {len(data_batch)} records to {stage_id}")
        except Exception as e:
            logger.error(
                f"[Checkpoint] Failed to save checkpoint for stage '{stage_id}'.\n"
                f"Reason: {str(e)}\n"
                f"Suggested fix: Verify disk space and write permissions for {checkpoint_path}"
            )
            raise

    def load(self, stage_id: str) -> list[dict]:
        """
        Loads all records from a stage's JSONL checkpoint file.
        Returns an empty list if the checkpoint does not exist.
        """
        checkpoint_path = self._get_checkpoint_path(stage_id)
        if not os.path.exists(checkpoint_path):
            return []

        records = []
        try:
            with open(checkpoint_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    stripped = line.strip()
                    if stripped:
                        try:
                            records.append(json.loads(stripped))
                        except json.JSONDecodeError as e:
                            logger.warning(
                                f"[Checkpoint] Skipping corrupted line {line_num} "
                                f"in {stage_id}: {str(e)}"
                            )
            logger.info(f"[Checkpoint] Loaded {len(records)} records from {stage_id}")
        except Exception as e:
            logger.error(
                f"[Checkpoint] Failed to load checkpoint for stage '{stage_id}'.\n"
                f"Reason: {str(e)}\n"
                f"Suggested fix: Delete corrupted checkpoint file and re-run the stage."
            )
            raise

        return records

    def is_complete(self, stage_id: str) -> bool:
        """Checks whether a stage has been marked as complete via its flag file."""
        flag_path = self._get_flag_path(stage_id)
        complete = os.path.exists(flag_path)
        if complete:
            logger.info(f"[Checkpoint] Stage '{stage_id}' already complete — skipping.")
        return complete

    def mark_complete(self, stage_id: str) -> None:
        """Writes a completion flag file for the given stage."""
        flag_path = self._get_flag_path(stage_id)
        try:
            with open(flag_path, 'w', encoding='utf-8') as f:
                f.write("complete\n")
            logger.info(f"[Checkpoint] Stage '{stage_id}' marked as complete.")
        except Exception as e:
            logger.error(
                f"[Checkpoint] Failed to mark stage '{stage_id}' as complete.\n"
                f"Reason: {str(e)}\n"
                f"Suggested fix: Verify write permissions for {flag_path}"
            )
            raise

    def clear(self, stage_id: str) -> None:
        """Deletes the JSONL checkpoint and flag file for a given stage."""
        checkpoint_path = self._get_checkpoint_path(stage_id)
        flag_path = self._get_flag_path(stage_id)

        for path in [checkpoint_path, flag_path]:
            if os.path.exists(path):
                try:
                    os.remove(path)
                    logger.debug(f"[Checkpoint] Removed: {path}")
                except Exception as e:
                    logger.warning(f"[Checkpoint] Could not remove {path}: {str(e)}")

    def clear_all(self) -> None:
        """Removes all checkpoint files. Called only after full pipeline success."""
        if os.path.exists(self.checkpoint_dir):
            for filename in os.listdir(self.checkpoint_dir):
                filepath = os.path.join(self.checkpoint_dir, filename)
                try:
                    if os.path.isfile(filepath):
                        os.remove(filepath)
                except Exception as e:
                    logger.warning(f"[Checkpoint] Could not remove {filepath}: {str(e)}")
            logger.info("[Checkpoint] All checkpoint files cleared.")
