import os
import json
import shutil
import subprocess
from datetime import datetime

import torch

from ..utils.logger import get_logger

logger = get_logger(__name__)


class Exporter:
    """
    Exports trained models to multiple formats: Safetensors, GGUF, and ONNX.
    Generates a model_card.md and copies relevant reports to the output directory.
    """

    @staticmethod
    def export(
        model,
        tokenizer,
        output_dir: str,
        formats: list[str] = None,
        model_info: dict = None,
        checkpoint_mgr=None,
    ) -> dict:
        """
        Exports the model to the requested formats and generates documentation.

        Args:
            model: The trained model.
            tokenizer: The model's tokenizer.
            output_dir: Base output directory.
            formats: List of export formats (e.g., ['safetensors', 'gguf', 'onnx']).
            model_info: Optional dict with model metadata for the model card.
            checkpoint_mgr: Optional CheckpointManager for resumability.

        Returns:
            Export manifest dict listing paths of all exported files.
        """
        if formats is None:
            formats = ["safetensors"]

        logger.info(f"[Exporter] Exporting model in formats: {formats}")
        manifest = {"exported_at": datetime.utcnow().isoformat(), "formats": {}}

        for fmt in formats:
            fmt_lower = fmt.lower().strip()

            if fmt_lower == "safetensors":
                path = Exporter._export_safetensors(model, tokenizer, output_dir)
                if path:
                    manifest["formats"]["safetensors"] = path

            elif fmt_lower == "gguf":
                path = Exporter._export_gguf(output_dir)
                if path:
                    manifest["formats"]["gguf"] = path

            elif fmt_lower == "onnx":
                path = Exporter._export_onnx(model, tokenizer, output_dir)
                if path:
                    manifest["formats"]["onnx"] = path

            else:
                logger.warning(f"[Exporter] Unknown export format: '{fmt}'. Skipping.")

        # Generate model card
        Exporter._generate_model_card(output_dir, model_info or {})

        # Generate training report
        Exporter._generate_training_report(output_dir, model_info or {}, manifest)

        # Save checkpoint
        if checkpoint_mgr:
            checkpoint_mgr.save("stage_12_export", [manifest])
            checkpoint_mgr.mark_complete("stage_12_export")

        logger.info(f"[Exporter] Export complete. Manifest: {manifest}")
        return manifest

    @staticmethod
    def _export_safetensors(model, tokenizer, output_dir: str) -> str | None:
        """Exports model weights in Safetensors format."""
        try:
            from safetensors.torch import save_file
        except ImportError:
            logger.warning(
                "[Exporter] safetensors not installed — skipping Safetensors export.\n"
                "Suggested fix: Run `pip install safetensors`"
            )
            return None

        try:
            # Save using HF's built-in safetensors support
            safetensors_dir = os.path.join(output_dir, "safetensors_model")
            os.makedirs(safetensors_dir, exist_ok=True)

            model.save_pretrained(safetensors_dir, safe_serialization=True)
            tokenizer.save_pretrained(safetensors_dir)

            logger.info(f"[Exporter] Safetensors model saved to {safetensors_dir}")
            return safetensors_dir

        except Exception as e:
            logger.error(f"[Exporter] Safetensors export failed: {str(e)}")
            return None

    @staticmethod
    def _export_gguf(output_dir: str) -> str | None:
        """
        Exports model to GGUF format via llama.cpp subprocess.
        Requires llama-cpp-python or llama.cpp convert script to be available.
        """
        gguf_path = os.path.join(output_dir, "trained_model.gguf")

        # Check if llama-cpp-python is available
        try:
            import llama_cpp
            logger.info("[Exporter] llama-cpp-python found.")
        except ImportError:
            logger.warning(
                "[Exporter] llama-cpp-python not installed — skipping GGUF export.\n"
                "Suggested fix: Install llama-cpp-python for GGUF conversion support."
            )
            return None

        # Try to find convert script
        convert_script = shutil.which("convert-hf-to-gguf")
        if convert_script is None:
            # Try common locations
            for candidate in ["convert-hf-to-gguf.py", "convert_hf_to_gguf.py"]:
                if os.path.exists(candidate):
                    convert_script = candidate
                    break

        if convert_script is None:
            logger.warning(
                "[Exporter] GGUF conversion script not found in PATH.\n"
                "Suggested fix: Install llama.cpp and ensure convert-hf-to-gguf is in PATH."
            )
            return None

        safetensors_dir = os.path.join(output_dir, "safetensors_model")
        if not os.path.exists(safetensors_dir):
            logger.warning("[Exporter] Safetensors model not found — export Safetensors first for GGUF conversion.")
            return None

        try:
            result = subprocess.run(
                [convert_script, safetensors_dir, "--outfile", gguf_path, "--outtype", "f16"],
                capture_output=True,
                text=True,
                timeout=600,
            )
            if result.returncode == 0:
                logger.info(f"[Exporter] GGUF model exported to {gguf_path}")
                return gguf_path
            else:
                logger.error(f"[Exporter] GGUF conversion failed: {result.stderr}")
                return None
        except Exception as e:
            logger.error(f"[Exporter] GGUF conversion error: {str(e)}")
            return None

    @staticmethod
    def _export_onnx(model, tokenizer, output_dir: str) -> str | None:
        """Exports model to ONNX format via torch.onnx.export."""
        try:
            import onnx  # noqa: F401
        except ImportError:
            logger.warning(
                "[Exporter] onnx not installed — skipping ONNX export.\n"
                "Suggested fix: Run `pip install onnx`"
            )
            return None

        onnx_path = os.path.join(output_dir, "trained_model.onnx")

        try:
            # Prepare dummy input
            dummy_text = "Hello, this is a test."
            inputs = tokenizer(dummy_text, return_tensors="pt")

            device = next(model.parameters()).device
            input_ids = inputs["input_ids"].to(device)

            model.eval()
            with torch.no_grad():
                torch.onnx.export(
                    model,
                    (input_ids,),
                    onnx_path,
                    opset_version=14,
                    input_names=["input_ids"],
                    output_names=["logits"],
                    dynamic_axes={
                        "input_ids": {0: "batch_size", 1: "sequence_length"},
                        "logits": {0: "batch_size", 1: "sequence_length"},
                    },
                )

            logger.info(f"[Exporter] ONNX model exported to {onnx_path}")
            return onnx_path

        except Exception as e:
            logger.warning(
                f"[Exporter] ONNX export failed: {str(e)}\n"
                f"Note: ONNX export may not be supported for all model architectures."
            )
            return None

    @staticmethod
    def _generate_model_card(output_dir: str, model_info: dict) -> None:
        """Generates a model_card.md with training metadata."""
        card_path = os.path.join(output_dir, "model_card.md")

        card_content = f"""# Model Card

## Overview
- **Framework**: Crineforge v0.2.7
- **Base Model**: {model_info.get('model_id', 'N/A')}
- **Training Mode**: {model_info.get('training_mode', 'lora')}
- **Date**: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}

## Training Details
- **Dataset**: {model_info.get('data_path', 'N/A')}
- **Dataset Format**: {model_info.get('dataset_format', 'instruction')}
- **Quality Threshold**: {model_info.get('quality_threshold', 0.4)}
- **Training Samples**: {model_info.get('train_samples', 'N/A')}

## Pipeline Stages Completed
{model_info.get('stages_summary', '- Full 12-stage pipeline')}

## Evaluation
- **Perplexity**: {model_info.get('perplexity', 'N/A')}
- **Avg Loss**: {model_info.get('avg_loss', 'N/A')}

## Export Formats
{model_info.get('export_summary', '- Safetensors')}

## Usage
```python
from crineforge import Trainer

trainer = Trainer()
trainer.connect_model("{model_info.get('model_id', 'path/to/model')}")
# Load and use the fine-tuned model
```

## License
See repository LICENSE file.
"""
        with open(card_path, 'w', encoding='utf-8') as f:
            f.write(card_content)
        logger.info(f"[Exporter] Model card generated at {card_path}")

    @staticmethod
    def _generate_training_report(output_dir: str, model_info: dict, manifest: dict) -> None:
        """Generates a training_report.json summarizing the full pipeline run."""
        report = {
            "model_id": model_info.get("model_id", "N/A"),
            "training_mode": model_info.get("training_mode", "lora"),
            "data_path": model_info.get("data_path", "N/A"),
            "quality_threshold": model_info.get("quality_threshold", 0.4),
            "completed_at": datetime.utcnow().isoformat(),
            "export_manifest": manifest,
        }

        report_path = os.path.join(output_dir, "training_report.json")
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        logger.info(f"[Exporter] Training report saved to {report_path}")
