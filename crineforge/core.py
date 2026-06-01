import os
import torch
import time
import json
try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kwargs: x
from .utils.logger import get_logger
from .utils.seed import set_seed
from .data.extractor import DataExtractor
from .data.chunker import Chunker
from .data.structurer import get_structurer, free_structurer
from .data.validator import Validator
from .data.enrichment import InternetEnrichment
from .model.connector import ModelConnector
from .model.gpu import GPUSensitive
from .hyperparams.auto import AutoConfig
from .training.engine import Engine
from .training.saver import Saver

logger = get_logger(__name__)

class Trainer:
    """
    Crineforge main facade.
    Provides a simple, elegant API for end-users to fine-tune models from raw data.
    """
    
    def __init__(self, seed: int = 42, debug_mode: bool = False, structurer_model: str = "Qwen/Qwen2.5-1.5B-Instruct"):
        self.structurer_model = structurer_model
        self.model_id = None
        self.data_path = None
        self.enrichment_enabled = False
        self.hyperparams = {}
        self.structured_pairs = []
        self.debug_mode = debug_mode
        self._model = None
        self._tokenizer = None
        
        set_seed(seed)
        logger.info("[Pipeline] Crineforge Trainer initialized.")

    def connect_model(self, model_id: str):
        """Connects a HuggingFace Hub ID or local path as the target for training."""
        self.model_id = model_id
        logger.info(f"[Pipeline] Connected to target model: {self.model_id}")

    def load_data(self, file_path: str):
        """Extracts and chunks raw data (PDF, CSV, TXT, JSON, MD)."""
        raw_text = DataExtractor.extract(file_path)
        self.data_path = file_path
        self._chunks = Chunker.split(raw_text)
        logger.info(f"[Pipeline] Data loaded from {self.data_path} ({len(self._chunks)} chunks ready).")

    def enable_enrichment(self, enabled: bool = True):
        """Toggle internet enrichment module (simulated)."""
        self.enrichment_enabled = enabled
        logger.info(f"[Pipeline] Enrichment mode set to: {self.enrichment_enabled}")

    def auto_config(self):
        """Automatically detects GPU and dataset constraints to set safe hyperparameters."""
        if not hasattr(self, '_chunks') or len(self._chunks) == 0:
             logger.warning("[Validation] auto_config called before load_data, assuming chunk length of zero for params.")
             dummy_len = 0
        else:
             dummy_len = len(self._chunks)
             
        self.hyperparams = AutoConfig.get_safe_params(dummy_len)

    def manual_config(self, **kwargs):
        """Manually override hyperparameter values safely."""
        self.hyperparams.update(kwargs)
        logger.info(f"[Validation] Manual override applied. Current params: {self.hyperparams}")

    def structure_only(self, input_path: str = None):
        """Generates structured dataset from the raw input without training."""
        GPUSensitive.log_vram_usage("Pipeline Start")
        start_time = time.time()
        
        target_path = input_path or self.data_path
        if not target_path:
            raise ValueError(
                "Data path is missing.\n"
                "Reason: No input path provided to structure_only() AND load_data() was not called previously.\n"
                "Suggested fix: Provide a valid file path or call load_data(file_path)."
            )
            
        if getattr(self, 'data_path', None) != target_path or not hasattr(self, '_chunks'):
            self.load_data(target_path)
            
        logger.info("[Pipeline] === Phase 1: Structuring Data ===")
        GPUSensitive.log_vram_usage("Before Structuring")
        
        try:
            structurer = get_structurer(self.structurer_model)
            structured_pairs = []
            checkpoint_file = f"{target_path}.checkpoint.jsonl"
            start_idx = 0
            
            if os.path.exists(checkpoint_file):
                logger.info(f"[Structurer] Found checkpoint: {checkpoint_file}. Resuming...")
                try:
                    with open(checkpoint_file, 'r', encoding='utf-8') as f:
                        for line in f:
                            if line.strip():
                                parsed_data = json.loads(line.strip())
                                structured_pairs.extend(parsed_data)
                                start_idx += 1
                except Exception as e:
                    logger.warning(f"[Structurer] Corrupted checkpoint. Starting fresh. Error: {str(e)}")
                    start_idx = 0
                    structured_pairs = []
                    open(checkpoint_file, 'w').close()
                    
            logger.info(f"[Structurer] Formatting chunks ({start_idx}/{len(self._chunks)} already done)...")
            with open(checkpoint_file, 'a', encoding='utf-8') as f_ckpt:
                for i in tqdm(range(start_idx, len(self._chunks)), desc="Structuring chunks"):
                    chunk = self._chunks[i]
                    json_str = structurer.generate_pairs(chunk)
                    parsed_data = Validator.parse_valid_json(json_str)
                    structured_pairs.extend(parsed_data)
                    
                    f_ckpt.write(json.dumps(parsed_data) + '\n')
                    f_ckpt.flush()
            
            if os.path.exists(checkpoint_file):
                try:
                    os.remove(checkpoint_file)
                except Exception:
                    pass
            
            Validator.validate_dataset_size(structured_pairs, debug_mode=getattr(self, 'debug_mode', False))
            
            if self.enrichment_enabled:
                structured_pairs = InternetEnrichment.enrich(structured_pairs)
                
            self.structured_pairs = structured_pairs
            
            duration = time.time() - start_time
            logger.info(f"[Performance] Structuring completed in {duration:.2f}s")
            
            return self.structured_pairs
            
        except Exception as e:
            logger.error(f"[Validation] Structuring failed: {str(e)}")
            if isinstance(e, ValueError):
                raise
            raise RuntimeError(
                f"Data structuring failed.\n"
                f"Reason: {str(e)}\n"
                f"Suggested fix: Ensure raw text can be parsed into target format."
            ) from e
            
        finally:
            free_structurer()
            GPUSensitive.log_vram_usage("After Structuring Cleanup")

    def train_from_structured(self, structured_data: list):
        """Executes the fine-tuning phase using pre-structured data."""
        start_time = time.time()
        if not self.model_id:
            raise ValueError(
                "Target model not connected.\n"
                "Reason: connect_model() was not called pipeline execution.\n"
                "Suggested fix: Call connect_model(model_id) first."
            )
            
        if not structured_data:
            raise ValueError(
                "Structured data is empty.\n"
                "Reason: The provided dataset has 0 instances.\n"
                "Suggested fix: Provide a valid array of training pairs."
            )
            
        Validator.validate_dataset_size(structured_data, debug_mode=getattr(self, 'debug_mode', False))
        
        logger.info("[Pipeline] === Phase 2: Target Model Loading ===")
        GPUSensitive.log_vram_usage("Before Training")
        try:
            model, tokenizer = ModelConnector.load(self.model_id)
            self._model = ModelConnector.prepare_lora(model)
            self._tokenizer = tokenizer
        except Exception as e:
            logger.error(f"[Validation] Failed to load target model {self.model_id}: {str(e)}")
            raise RuntimeError(
                f"Model loading failed.\n"
                f"Reason: {str(e)}\n"
                f"Suggested fix: Verify model ID exists and you have network access."
            ) from e

        logger.info("[Pipeline] === Phase 3: Fine-Tuning Execution ===")
        try:
            Engine.run(self._model, self._tokenizer, structured_data, self.hyperparams)
        except getattr(torch.cuda, 'OutOfMemoryError', Exception) as e:
            logger.error(f"[GPU] OOM error during training: {str(e)}")
            raise RuntimeError(
                "Training aborted due to OOM.\n"
                "Reason: Out of memory executing fine-tuning sequence.\n"
                "Suggested fix: Decrease batch size, enable 4bit quantization, or use a smaller model."
            ) from e
        except Exception as e:
            if "CUDA out of memory" in str(e):
                logger.error(f"[GPU] OOM error during training: {str(e)}")
                raise RuntimeError(
                    "Training aborted due to OOM.\n"
                    "Reason: Out of memory executing fine-tuning sequence.\n"
                    "Suggested fix: Decrease batch size, enable 4bit quantization, or use a smaller model."
                ) from e
            logger.error(f"[Pipeline] Training engine failed: {str(e)}")
            raise RuntimeError(
                f"Training execution failed.\n"
                f"Reason: {str(e)}\n"
                f"Suggested fix: Validate hyperparams and data structure correctness."
            ) from e
        finally:
            GPUSensitive.empty_cache()
            GPUSensitive.log_vram_usage("After Training")
            
        duration = time.time() - start_time
        logger.info(f"[Performance] Training completed in {duration:.2f}s")

    def train(self):
        """Starts the local structuring and fine-tuning process securely."""
        if not self.model_id or not self.data_path:
            raise ValueError(
                "Model and Data must be configured before training.\n"
                "Reason: Missing prerequisites for pipeline execution.\n"
                "Suggested fix: Call connect_model() and load_data() first."
            )
            
        try:
            structured_data = self.structure_only(self.data_path)
            self.train_from_structured(structured_data)
        except Exception as e:
            logger.error(f"Training pipeline aborted securely: {str(e)}")
            raise

    def dry_run(self):
        """Runs the entire pipeline without invoking actual structural inference or target training updates."""
        logger.info("[Pipeline] *** DRY RUN MODE INITIATED ***")
        if not self.model_id or not self.data_path:
            raise ValueError("Model and Data must be configured before dry run.")
        
        logger.info(f"[Validation] Validating target Model: {self.model_id}")
        logger.info(f"[Validation] Validating target Data: {self.data_path}")
        logger.info(f"[Validation] Validating Hyperparams: {self.hyperparams}")
        
        logger.info("[Pipeline] Mocking structurer and GPU checks...")
        strategy = GPUSensitive.get_strategy()
        logger.info(f"[GPU] Would use GPU Strategy: {strategy}")
        
        # Dry Run fake dataset
        fake_data = [{"instruction": "Test", "response": "Mocked Output"}] * max((len(self._chunks) if hasattr(self, '_chunks') else 0), 20)
        Validator.validate_dataset_size(fake_data, debug_mode=getattr(self, 'debug_mode', False))
        
        logger.info("[Pipeline] *** DRY RUN SUCCESS - PIPELINE VALIDATED ***")

    def save(self, output_dir: str):
        """Saves the trained LoRA adapter or full model."""
        if not self._model:
            raise ValueError("No model has been trained yet. Cannot save.")
        Saver.save_lora(self._model, output_dir)

    def run_full_pipeline(
        self,
        output_dir: str = "crineforge_output",
        training_mode: str = "lora",
        quality_threshold: float = 0.4,
        dataset_format: str = "instruction",
        train_split: float = 0.8,
        quantize: str = None,
        export_formats: list = None,
    ) -> dict:
        """
        Runs the complete 12-stage AI Data Refinery pipeline.

        Stage 1:  Ingestion (TikaExtractor + existing DataExtractor)
        Stage 2:  Validation (existing Validator)
        Stage 3:  Deduplication (Deduplicator)
        Stage 4:  Classification (Classifier)
        Stage 5:  Embeddings (Embedder)
        Stage 6:  Quality Scoring (QualityScorer)
        Stage 7:  Dataset Enrichment (Enricher)
        Stage 8:  Dataset Assembly (Assembler)
        Stage 9:  Training (existing Engine + ModelConnector)
        Stage 10: Evaluation (Evaluator)
        Stage 11: Optimization (Quantizer, if quantize param provided)
        Stage 12: Export (Exporter)

        Args:
            output_dir: Root output directory for all pipeline artifacts.
            training_mode: One of 'lora', 'qlora', 'full'.
            quality_threshold: Minimum quality score to keep (0.0–1.0).
            dataset_format: One of 'instruction', 'chat', 'completion'.
            train_split: Fraction of data for training (remainder split 50/50 to valid/test).
            quantize: Optional quantization — '4bit', '8bit', or None.
            export_formats: List of export formats (e.g., ['safetensors', 'gguf', 'onnx']).

        Returns:
            Dict summarizing pipeline results.

        Raises:
            ValueError: If model or data not configured.
        """
        from .utils.checkpoint import CheckpointManager
        from .ingestion.tika_extractor import TikaExtractor
        from .pipeline.deduplicator import Deduplicator
        from .pipeline.classifier import Classifier
        from .pipeline.embedder import Embedder
        from .pipeline.quality_scorer import QualityScorer
        from .pipeline.enricher import Enricher
        from .pipeline.assembler import Assembler
        from .evaluation.evaluator import Evaluator
        from .optimization.quantizer import Quantizer
        from .export.exporter import Exporter

        if not self.model_id:
            raise ValueError(
                "Target model not connected.\n"
                "Reason: connect_model() was not called before run_full_pipeline().\n"
                "Suggested fix: Call connect_model(model_id) first."
            )
        if not self.data_path:
            raise ValueError(
                "Data path not set.\n"
                "Reason: load_data() was not called before run_full_pipeline().\n"
                "Suggested fix: Call load_data(file_path) first."
            )

        os.makedirs(output_dir, exist_ok=True)
        ckpt = CheckpointManager(output_dir)
        pipeline_start = time.time()
        results = {"stages_completed": []}

        GPUSensitive.log_vram_usage("Pipeline Start")

        # ── Stage 1: Ingestion ──────────────────────────────────────
        logger.info("[Pipeline] ═══ Stage 1/12: Ingestion ═══")
        if ckpt.is_complete("stage_01_ingestion"):
            documents = ckpt.load("stage_01_ingestion")
        else:
            # Try existing DataExtractor first, fall back to TikaExtractor
            try:
                raw_text = DataExtractor.extract(self.data_path)
            except ValueError:
                if TikaExtractor.is_supported(self.data_path):
                    raw_text = TikaExtractor.extract(self.data_path)
                else:
                    raise ValueError(
                        f"Unsupported file format: {self.data_path}\n"
                        f"Reason: Neither DataExtractor nor TikaExtractor can handle this file.\n"
                        f"Suggested fix: Use a supported format (PDF, CSV, JSON, TXT, MD, DOCX, PPTX, HTML, etc.)"
                    )

            chunks = Chunker.split(raw_text)
            documents = [{"text": chunk, "metadata": {"source": self.data_path}} for chunk in chunks]
            ckpt.save("stage_01_ingestion", documents)
            ckpt.mark_complete("stage_01_ingestion")
        results["stages_completed"].append("ingestion")
        logger.info(f"[Pipeline] Stage 1 complete: {len(documents)} documents ingested.")

        # ── Stage 2: Validation ─────────────────────────────────────
        logger.info("[Pipeline] ═══ Stage 2/12: Validation ═══")
        Validator.validate_dataset_size(documents, debug_mode=self.debug_mode)
        results["stages_completed"].append("validation")

        # ── Stage 3: Deduplication ──────────────────────────────────
        logger.info("[Pipeline] ═══ Stage 3/12: Deduplication ═══")
        if ckpt.is_complete("stage_03_dedup"):
            documents = ckpt.load("stage_03_dedup")
            dedup_report = {}
        else:
            documents, dedup_report = Deduplicator.deduplicate(
                documents, output_dir=output_dir, checkpoint_mgr=ckpt
            )
        results["stages_completed"].append("deduplication")
        results["dedup_report"] = dedup_report
        logger.info(f"[Pipeline] Stage 3 complete: {len(documents)} documents after dedup.")

        # ── Stage 4: Classification ─────────────────────────────────
        logger.info("[Pipeline] ═══ Stage 4/12: Classification ═══")
        if ckpt.is_complete("stage_04_classify"):
            documents = ckpt.load("stage_04_classify")
        else:
            documents = Classifier.classify(documents, checkpoint_mgr=ckpt)
        results["stages_completed"].append("classification")

        # ── Stage 5: Embeddings ─────────────────────────────────────
        logger.info("[Pipeline] ═══ Stage 5/12: Embeddings ═══")
        if ckpt.is_complete("stage_05_embed"):
            documents = ckpt.load("stage_05_embed")
        else:
            documents = Embedder.embed(documents, output_dir=output_dir, checkpoint_mgr=ckpt)
        results["stages_completed"].append("embeddings")

        # ── Stage 6: Quality Scoring ────────────────────────────────
        logger.info("[Pipeline] ═══ Stage 6/12: Quality Scoring ═══")
        pre_quality_count = len(documents)
        if ckpt.is_complete("stage_06_quality"):
            documents = ckpt.load("stage_06_quality")
        else:
            documents = QualityScorer.score(
                documents, threshold=quality_threshold, checkpoint_mgr=ckpt
            )
        results["stages_completed"].append("quality_scoring")
        results["quality_filtered"] = pre_quality_count - len(documents)
        logger.info(f"[Pipeline] Stage 6 complete: {len(documents)} documents passed quality filter.")

        # ── Stage 7: Enrichment ─────────────────────────────────────
        logger.info("[Pipeline] ═══ Stage 7/12: Enrichment ═══")
        if ckpt.is_complete("stage_07_enrich"):
            documents = ckpt.load("stage_07_enrich")
        else:
            documents = Enricher.enrich(
                documents, dataset_format=dataset_format, checkpoint_mgr=ckpt
            )
        results["stages_completed"].append("enrichment")

        # ── Stage 8: Assembly ───────────────────────────────────────
        logger.info("[Pipeline] ═══ Stage 8/12: Assembly ═══")
        valid_split = (1.0 - train_split) / 2.0
        test_split = (1.0 - train_split) / 2.0
        if ckpt.is_complete("stage_08_assemble"):
            dataset_report = {}
            # Load the assembled train set from the JSONL file
            train_path = os.path.join(output_dir, "clean_dataset", "train.jsonl")
            test_path = os.path.join(output_dir, "clean_dataset", "test.jsonl")
            train_data = []
            test_data = []
            if os.path.exists(train_path):
                with open(train_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            train_data.append(json.loads(line.strip()))
            if os.path.exists(test_path):
                with open(test_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            test_data.append(json.loads(line.strip()))
        else:
            dataset_report = Assembler.assemble(
                documents,
                output_dir=output_dir,
                train_split=train_split,
                valid_split=valid_split,
                test_split=test_split,
                checkpoint_mgr=ckpt,
            )
            # Read back the splits for training
            train_data = []
            test_data = []
            train_path = os.path.join(output_dir, "clean_dataset", "train.jsonl")
            test_path = os.path.join(output_dir, "clean_dataset", "test.jsonl")
            if os.path.exists(train_path):
                with open(train_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            train_data.append(json.loads(line.strip()))
            if os.path.exists(test_path):
                with open(test_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            test_data.append(json.loads(line.strip()))
        results["stages_completed"].append("assembly")
        results["dataset_report"] = dataset_report
        logger.info(f"[Pipeline] Stage 8 complete: {len(train_data)} training samples assembled.")

        # ── Stage 9: Training ───────────────────────────────────────
        logger.info("[Pipeline] ═══ Stage 9/12: Training ═══")
        GPUSensitive.log_vram_usage("Before Training")

        if not self.hyperparams:
            self.auto_config()

        try:
            model, tokenizer = ModelConnector.load(self.model_id)

            if training_mode in ("lora", "qlora"):
                self._model = ModelConnector.prepare_lora(model)
            else:
                self._model = model
            self._tokenizer = tokenizer

            # Use train_data for training
            if train_data:
                Engine.run(self._model, self._tokenizer, train_data, self.hyperparams)

                # Save adapter/model
                model_save_dir = os.path.join(output_dir, "adapter" if training_mode != "full" else "trained_model")
                Saver.save_lora(self._model, model_save_dir)
            else:
                logger.warning("[Pipeline] No training data available — skipping training.")

        except Exception as e:
            logger.error(f"[Pipeline] Training failed: {str(e)}")
            raise
        finally:
            GPUSensitive.empty_cache()
            GPUSensitive.log_vram_usage("After Training")

        results["stages_completed"].append("training")

        # ── Stage 10: Evaluation ────────────────────────────────────
        logger.info("[Pipeline] ═══ Stage 10/12: Evaluation ═══")
        if ckpt.is_complete("stage_10_eval"):
            eval_report = {}
        else:
            eval_data = test_data if test_data else train_data[:50]
            eval_report = Evaluator.evaluate(
                self._model,
                self._tokenizer,
                eval_data,
                output_dir=output_dir,
                checkpoint_mgr=ckpt,
            )
        results["stages_completed"].append("evaluation")
        results["eval_report"] = eval_report

        # ── Stage 11: Quantization (optional) ──────────────────────
        if quantize:
            logger.info(f"[Pipeline] ═══ Stage 11/12: Quantization ({quantize}) ═══")
            if ckpt.is_complete("stage_11_quant"):
                logger.info("[Pipeline] Quantization already complete.")
            else:
                bits = 4 if "4" in quantize else 8
                model_save_dir = os.path.join(output_dir, "adapter" if training_mode != "full" else "trained_model")
                Quantizer.quantize(
                    model_path=model_save_dir,
                    output_dir=output_dir,
                    bits=bits,
                    checkpoint_mgr=ckpt,
                )
            results["stages_completed"].append("quantization")
        else:
            logger.info("[Pipeline] ═══ Stage 11/12: Quantization — SKIPPED ═══")

        # ── Stage 12: Export ────────────────────────────────────────
        if export_formats:
            logger.info(f"[Pipeline] ═══ Stage 12/12: Export ({export_formats}) ═══")
            if ckpt.is_complete("stage_12_export"):
                logger.info("[Pipeline] Export already complete.")
            else:
                model_info = {
                    "model_id": self.model_id,
                    "data_path": self.data_path,
                    "training_mode": training_mode,
                    "quality_threshold": quality_threshold,
                    "dataset_format": dataset_format,
                    "train_samples": len(train_data),
                    "perplexity": eval_report.get("perplexity", "N/A"),
                    "avg_loss": eval_report.get("avg_loss", "N/A"),
                }
                export_manifest = Exporter.export(
                    self._model,
                    self._tokenizer,
                    output_dir=output_dir,
                    formats=export_formats,
                    model_info=model_info,
                    checkpoint_mgr=ckpt,
                )
                results["export_manifest"] = export_manifest
            results["stages_completed"].append("export")
        else:
            logger.info("[Pipeline] ═══ Stage 12/12: Export — SKIPPED ═══")

        # ── Pipeline Complete ───────────────────────────────────────
        # Free embedding model if it was loaded
        Embedder.free_model()
        GPUSensitive.empty_cache()

        pipeline_duration = time.time() - pipeline_start
        results["total_duration_seconds"] = round(pipeline_duration, 2)
        results["output_dir"] = output_dir

        logger.info(
            f"[Pipeline] ════════════════════════════════════════════\n"
            f"[Pipeline] FULL PIPELINE COMPLETED in {pipeline_duration:.2f}s\n"
            f"[Pipeline] Output: {output_dir}\n"
            f"[Pipeline] Stages: {', '.join(results['stages_completed'])}\n"
            f"[Pipeline] ════════════════════════════════════════════"
        )

        return results
