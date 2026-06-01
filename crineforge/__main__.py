import argparse
import sys
import logging
from .core import Trainer

def main():
    parser = argparse.ArgumentParser(description="Crineforge - Define and run AI training jobs.")
    
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train a model using local data")
    train_parser.add_argument("--model", type=str, required=True, help="HF model ID or local path")
    train_parser.add_argument("--data", type=str, required=True, help="Path to the dataset (PDF, CSV, TXT, JSON, MD)")
    train_parser.add_argument("--output", type=str, default="output_model", help="Output directory")
    train_parser.add_argument("--dry-run", action="store_true", help="Run the pipeline without training")

    # Refine command — full 12-stage AI Data Refinery pipeline
    refine_parser = subparsers.add_parser(
        "refine",
        help="Run the full AI Data Refinery pipeline (ingest → deduplicate → classify → embed → score → enrich → assemble → train → evaluate → quantize → export)"
    )
    refine_parser.add_argument("--model", type=str, required=True, help="HF model ID or local path")
    refine_parser.add_argument("--data", type=str, required=True, help="Path to the dataset (any supported format)")
    refine_parser.add_argument("--output", type=str, default="crineforge_output", help="Output directory")
    refine_parser.add_argument(
        "--mode", type=str, default="lora",
        choices=["lora", "qlora", "full"],
        help="Training mode (default: lora)"
    )
    refine_parser.add_argument(
        "--quality-threshold", type=float, default=0.4,
        help="Minimum quality score to keep samples (default: 0.4)"
    )
    refine_parser.add_argument(
        "--format", type=str, default="instruction",
        choices=["instruction", "chat", "completion"],
        help="Dataset format (default: instruction)"
    )
    refine_parser.add_argument(
        "--train-split", type=float, default=0.8,
        help="Fraction of data for training (default: 0.8)"
    )
    refine_parser.add_argument(
        "--quantize", type=str, default=None,
        choices=["4bit", "8bit"],
        help="Quantize the trained model (optional)"
    )
    refine_parser.add_argument(
        "--export", type=str, default=None,
        help="Comma-separated export formats, e.g. 'safetensors,gguf,onnx'"
    )
    refine_parser.add_argument(
        "--api", action="store_true",
        help="Start the FastAPI server instead of running the pipeline directly"
    )
    refine_parser.add_argument(
        "--port", type=int, default=8000,
        help="Port for the API server (default: 8000)"
    )
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    logger = logging.getLogger("crineforge.cli")
    
    if args.command == "train":
        try:
            logger.info(f"Starting Crineforge with model={args.model} and data={args.data}")
            trainer = Trainer()
            trainer.connect_model(args.model)
            trainer.load_data(args.data)
            trainer.auto_config()
            
            if args.dry_run:
                logger.info("Executing dry run...")
                trainer.dry_run()
            else:
                logger.info("Starting training...")
                trainer.train()
                trainer.save(args.output)
                logger.info(f"Training complete. Model saved to {args.output}")
                
        except ValueError as e:
            logger.error(f"[User Error] {e}")
            sys.exit(1)
        except RuntimeError as e:
            logger.error(f"[Runtime Error] {e}")
            sys.exit(1)
        except Exception as e:
            logger.error(f"[Unexpected Error] {e}")
            raise

    elif args.command == "refine":
        # Handle --api flag: start FastAPI server instead of running pipeline
        if args.api:
            try:
                from .api.server import start_api
                logger.info(f"Starting Crineforge API server on port {args.port}...")
                start_api(host="0.0.0.0", port=args.port)
            except ImportError as e:
                logger.error(f"[Import Error] {e}")
                sys.exit(1)
            return

        # Run the full 12-stage pipeline
        try:
            logger.info(
                f"Starting Crineforge Refinery pipeline with "
                f"model={args.model}, data={args.data}, mode={args.mode}"
            )

            trainer = Trainer()
            trainer.connect_model(args.model)
            trainer.load_data(args.data)
            trainer.auto_config()

            export_formats = None
            if args.export:
                export_formats = [fmt.strip() for fmt in args.export.split(",")]

            results = trainer.run_full_pipeline(
                output_dir=args.output,
                training_mode=args.mode,
                quality_threshold=args.quality_threshold,
                dataset_format=args.format,
                train_split=args.train_split,
                quantize=args.quantize,
                export_formats=export_formats,
            )

            logger.info(f"Refinery pipeline complete. Results in {args.output}")
            logger.info(f"Stages completed: {', '.join(results.get('stages_completed', []))}")
            logger.info(f"Total duration: {results.get('total_duration_seconds', 0):.2f}s")

        except ValueError as e:
            logger.error(f"[User Error] {e}")
            sys.exit(1)
        except RuntimeError as e:
            logger.error(f"[Runtime Error] {e}")
            sys.exit(1)
        except KeyboardInterrupt:
            logger.warning("[Pipeline] Interrupted by user (Ctrl+C). Checkpoints saved — resume with same command.")
            sys.exit(130)
        except Exception as e:
            logger.error(f"[Unexpected Error] {e}")
            raise

if __name__ == "__main__":
    main()

