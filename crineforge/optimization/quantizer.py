import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from ..model.gpu import GPUSensitive
from ..utils.logger import get_logger

logger = get_logger(__name__)


class Quantizer:
    """
    Quantizes models to 4-bit or 8-bit precision using bitsandbytes.
    Saves the quantized model to a dedicated output directory.
    """

    @staticmethod
    def quantize(
        model_path: str,
        output_dir: str,
        bits: int = 4,
        checkpoint_mgr=None,
    ) -> str:
        """
        Quantizes a model to the specified bit precision.

        Args:
            model_path: Path to the model (HF Hub ID or local directory).
            output_dir: Base output directory.
            bits: Quantization precision — 4 or 8.
            checkpoint_mgr: Optional CheckpointManager for resumability.

        Returns:
            Path to the quantized model directory.

        Raises:
            ValueError: If bits is not 4 or 8.
        """
        if bits not in (4, 8):
            raise ValueError(
                f"Invalid quantization precision: {bits}-bit.\n"
                f"Reason: Only 4-bit and 8-bit quantization are supported.\n"
                f"Suggested fix: Use --quantize 4bit or --quantize 8bit."
            )

        logger.info(f"[Quantizer] Starting {bits}-bit quantization for: {model_path}")
        GPUSensitive.log_vram_usage("Before Quantization")

        # Configure quantization
        if bits == 4:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        else:  # 8-bit
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
            )

        # Determine device strategy
        strategy = GPUSensitive.get_strategy()
        device_map = "auto" if strategy["device"] == "cuda" else "cpu"

        hf_token = os.environ.get("HF_TOKEN")

        try:
            logger.info(f"[Quantizer] Loading model with {bits}-bit quantization...")
            tokenizer = AutoTokenizer.from_pretrained(
                model_path, trust_remote_code=True, token=hf_token
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=bnb_config,
                device_map=device_map,
                trust_remote_code=True,
                token=hf_token,
            )

            # Save quantized model
            quant_dir = os.path.join(output_dir, "quantized_models", f"model_{bits}bit")
            os.makedirs(quant_dir, exist_ok=True)

            model.save_pretrained(quant_dir)
            tokenizer.save_pretrained(quant_dir)
            logger.info(f"[Quantizer] {bits}-bit quantized model saved to {quant_dir}")

            # Save checkpoint
            if checkpoint_mgr:
                checkpoint_mgr.save("stage_11_quant", [{"bits": bits, "path": quant_dir}])
                checkpoint_mgr.mark_complete("stage_11_quant")

            GPUSensitive.log_vram_usage("After Quantization")
            GPUSensitive.empty_cache()

            return quant_dir

        except Exception as e:
            logger.error(
                f"[Quantizer] {bits}-bit quantization failed.\n"
                f"Reason: {str(e)}\n"
                f"Suggested fix: Ensure bitsandbytes is installed and GPU is available."
            )
            raise RuntimeError(
                f"Model quantization failed.\n"
                f"Reason: {str(e)}\n"
                f"Suggested fix: Verify CUDA availability and bitsandbytes installation."
            ) from e
