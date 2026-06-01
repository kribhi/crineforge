import os
import json
import math

import torch

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kwargs: x

from ..model.gpu import GPUSensitive
from ..utils.logger import get_logger

logger = get_logger(__name__)


class Evaluator:
    """
    Evaluates a trained model on loss, perplexity, instruction-following
    quality (heuristic), and JSON generation quality (parse check).
    """

    @staticmethod
    def evaluate(
        model,
        tokenizer,
        eval_dataset: list[dict],
        output_dir: str,
        max_samples: int = 100,
        checkpoint_mgr=None,
    ) -> dict:
        """
        Runs evaluation and generates a report.

        Args:
            model: The trained model (HF / PEFT).
            tokenizer: The tokenizer for the model.
            eval_dataset: List of document dicts with 'instruction' and 'response' keys.
            output_dir: Directory to save evaluation_report.json.
            max_samples: Maximum samples to evaluate for performance.
            checkpoint_mgr: Optional CheckpointManager for resumability.

        Returns:
            Evaluation report dict.
        """
        if not eval_dataset:
            logger.warning("[Eval] No evaluation data provided. Skipping evaluation.")
            return {"status": "skipped", "reason": "no_eval_data"}

        logger.info(f"[Eval] Starting evaluation on {min(len(eval_dataset), max_samples)} samples...")
        GPUSensitive.log_vram_usage("Before Evaluation")

        # Limit samples for performance
        samples = eval_dataset[:max_samples]

        # Prepare evaluation texts
        texts = []
        for doc in samples:
            instruction = doc.get("instruction", "")
            response = doc.get("response", "")
            if instruction and response:
                texts.append(f"Instruction: {instruction}\nResponse: {response}")
            elif doc.get("text"):
                texts.append(doc["text"])

        if not texts:
            logger.warning("[Eval] No valid text samples found for evaluation.")
            return {"status": "skipped", "reason": "no_valid_texts"}

        report = {}

        # 1. Compute loss and perplexity
        try:
            avg_loss, perplexity = Evaluator._compute_perplexity(model, tokenizer, texts)
            report["avg_loss"] = round(avg_loss, 6)
            report["perplexity"] = round(perplexity, 4)
            logger.info(f"[Eval] Loss: {avg_loss:.6f}, Perplexity: {perplexity:.4f}")
        except Exception as e:
            logger.warning(f"[Eval] Perplexity computation failed: {str(e)}")
            report["avg_loss"] = None
            report["perplexity"] = None

        # 2. Instruction-following quality
        try:
            instruction_score = Evaluator._check_instruction_following(
                model, tokenizer, samples[:20]
            )
            report["instruction_following_score"] = round(instruction_score, 4)
            logger.info(f"[Eval] Instruction-following score: {instruction_score:.4f}")
        except Exception as e:
            logger.warning(f"[Eval] Instruction-following check failed: {str(e)}")
            report["instruction_following_score"] = None

        # 3. JSON generation quality
        try:
            json_score = Evaluator._check_json_quality(model, tokenizer, samples[:10])
            report["json_generation_score"] = round(json_score, 4)
            logger.info(f"[Eval] JSON generation score: {json_score:.4f}")
        except Exception as e:
            logger.warning(f"[Eval] JSON quality check failed: {str(e)}")
            report["json_generation_score"] = None

        report["total_samples_evaluated"] = len(texts)
        report["status"] = "completed"

        # Write report
        eval_dir = os.path.join(output_dir, "evaluation_reports")
        os.makedirs(eval_dir, exist_ok=True)
        report_path = os.path.join(eval_dir, "evaluation_report.json")
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        logger.info(f"[Eval] Evaluation report saved to {report_path}")

        # Save checkpoint
        if checkpoint_mgr:
            checkpoint_mgr.save("stage_10_eval", [report])
            checkpoint_mgr.mark_complete("stage_10_eval")

        GPUSensitive.log_vram_usage("After Evaluation")
        return report

    @staticmethod
    def _compute_perplexity(model, tokenizer, texts: list[str]) -> tuple[float, float]:
        """
        Computes average loss and perplexity over the given texts.
        """
        model.eval()
        total_loss = 0.0
        total_count = 0

        device = next(model.parameters()).device

        with torch.no_grad():
            for text in tqdm(texts, desc="Computing perplexity"):
                encodings = tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                ).to(device)

                outputs = model(**encodings, labels=encodings["input_ids"])
                loss = outputs.loss

                if loss is not None:
                    total_loss += loss.item()
                    total_count += 1

        if total_count == 0:
            return 0.0, float('inf')

        avg_loss = total_loss / total_count
        perplexity = math.exp(min(avg_loss, 100))  # Cap to prevent overflow
        return avg_loss, perplexity

    @staticmethod
    def _check_instruction_following(
        model, tokenizer, samples: list[dict]
    ) -> float:
        """
        Heuristic check: does the model output address the instruction?
        Generates a response and checks if key terms from the instruction appear.
        """
        if not samples:
            return 0.0

        model.eval()
        device = next(model.parameters()).device
        score_sum = 0.0
        count = 0

        for doc in samples:
            instruction = doc.get("instruction", "")
            if not instruction:
                continue

            prompt = f"Instruction: {instruction}\nResponse:"
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=256,
            ).to(device)

            try:
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=128,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                response = tokenizer.decode(
                    outputs[0][inputs.input_ids.shape[-1]:],
                    skip_special_tokens=True,
                ).strip()

                # Heuristic: check if key instruction words appear in response
                instruction_words = set(instruction.lower().split())
                response_words = set(response.lower().split())

                # Remove common stopwords
                stopwords = {"the", "a", "an", "is", "are", "was", "were", "in", "on", "at", "to", "for"}
                instruction_words -= stopwords
                response_words -= stopwords

                if instruction_words:
                    overlap = len(instruction_words & response_words) / len(instruction_words)
                else:
                    overlap = 0.5

                # Also check response isn't empty
                if len(response) > 10:
                    score_sum += min(overlap + 0.3, 1.0)
                else:
                    score_sum += 0.1

                count += 1

            except Exception as e:
                logger.debug(f"[Eval] Generation failed for sample: {str(e)}")
                count += 1

        return score_sum / max(count, 1)

    @staticmethod
    def _check_json_quality(model, tokenizer, samples: list[dict]) -> float:
        """
        Checks if the model can produce valid JSON output when prompted.
        """
        if not samples:
            return 0.0

        model.eval()
        device = next(model.parameters()).device
        valid_count = 0
        total = 0

        for doc in samples:
            prompt = (
                "Generate a JSON object with keys 'summary' and 'category' "
                f"for the following text: {doc.get('instruction', doc.get('text', ''))[:200]}\n"
                "Output valid JSON only:\n"
            )

            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=256,
            ).to(device)

            try:
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=128,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                response = tokenizer.decode(
                    outputs[0][inputs.input_ids.shape[-1]:],
                    skip_special_tokens=True,
                ).strip()

                # Try to parse JSON
                json.loads(response)
                valid_count += 1
            except (json.JSONDecodeError, Exception):
                pass

            total += 1

        return valid_count / max(total, 1)
