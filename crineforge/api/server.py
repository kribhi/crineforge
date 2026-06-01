import os
import json
import uuid
import threading
from datetime import datetime

from ..utils.logger import get_logger

logger = get_logger(__name__)

# Job storage (in-memory + persisted to disk)
_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()


def _create_app():
    """Creates the FastAPI application. Lazy import to avoid hard dependency."""
    try:
        from fastapi import FastAPI, HTTPException
        from pydantic import BaseModel
    except ImportError:
        raise ImportError(
            "FastAPI is not installed.\n"
            "Reason: The API server requires FastAPI and uvicorn.\n"
            "Suggested fix: Run `pip install fastapi uvicorn`"
        )

    app = FastAPI(
        title="Crineforge API",
        description="AI Data Refinery and Automated Model Training Platform",
        version="0.2.7",
    )

    class PipelineRequest(BaseModel):
        """Request body for starting a pipeline run."""
        model_id: str
        data_path: str
        output_dir: str = "crineforge_output"
        training_mode: str = "lora"
        quality_threshold: float = 0.4
        dataset_format: str = "instruction"
        train_split: float = 0.8
        quantize: str | None = None
        export_formats: list[str] | None = None

    class JobInfo(BaseModel):
        """Response model for job information."""
        job_id: str
        status: str
        stage: str
        progress: float
        created_at: str
        updated_at: str
        error: str | None = None

    def _run_pipeline_job(job_id: str, request: PipelineRequest) -> None:
        """Runs the full pipeline in a background thread."""
        try:
            _update_job(job_id, status="running", stage="initializing", progress=0.0)

            from ..core import Trainer

            trainer = Trainer()
            trainer.connect_model(request.model_id)

            _update_job(job_id, stage="loading_data", progress=5.0)
            trainer.load_data(request.data_path)
            trainer.auto_config()

            _update_job(job_id, stage="pipeline_running", progress=10.0)

            export_formats = request.export_formats
            result = trainer.run_full_pipeline(
                output_dir=request.output_dir,
                training_mode=request.training_mode,
                quality_threshold=request.quality_threshold,
                dataset_format=request.dataset_format,
                train_split=request.train_split,
                quantize=request.quantize,
                export_formats=export_formats,
            )

            _update_job(
                job_id,
                status="completed",
                stage="finished",
                progress=100.0,
                result=result,
            )
            logger.info(f"[API] Job {job_id} completed successfully.")

        except Exception as e:
            logger.error(f"[API] Job {job_id} failed: {str(e)}")
            _update_job(job_id, status="failed", stage="error", error=str(e))

    def _update_job(job_id: str, **kwargs) -> None:
        """Updates job state in memory and persists to disk."""
        with _jobs_lock:
            if job_id in _jobs:
                _jobs[job_id].update(kwargs)
                _jobs[job_id]["updated_at"] = datetime.utcnow().isoformat()

                # Persist to disk
                output_dir = _jobs[job_id].get("output_dir", "crineforge_output")
                jobs_dir = os.path.join(output_dir, "jobs")
                os.makedirs(jobs_dir, exist_ok=True)
                job_path = os.path.join(jobs_dir, f"{job_id}.json")

                # Filter non-serializable fields
                serializable = {
                    k: v for k, v in _jobs[job_id].items()
                    if isinstance(v, (str, int, float, bool, list, dict, type(None)))
                }
                try:
                    with open(job_path, 'w', encoding='utf-8') as f:
                        json.dump(serializable, f, indent=2, ensure_ascii=False)
                except Exception as e:
                    logger.warning(f"[API] Failed to persist job state: {str(e)}")

    @app.post("/pipeline/run")
    def run_pipeline(request: PipelineRequest) -> dict:
        """Starts a new pipeline run in the background."""
        job_id = str(uuid.uuid4())[:8]
        now = datetime.utcnow().isoformat()

        with _jobs_lock:
            _jobs[job_id] = {
                "job_id": job_id,
                "status": "queued",
                "stage": "initializing",
                "progress": 0.0,
                "created_at": now,
                "updated_at": now,
                "model_id": request.model_id,
                "data_path": request.data_path,
                "output_dir": request.output_dir,
                "error": None,
            }

        thread = threading.Thread(
            target=_run_pipeline_job,
            args=(job_id, request),
            daemon=True,
        )
        thread.start()

        logger.info(f"[API] Pipeline job {job_id} started for model={request.model_id}")
        return {"job_id": job_id, "status": "queued"}

    @app.get("/pipeline/status/{job_id}")
    def get_pipeline_status(job_id: str) -> dict:
        """Returns the current status of a pipeline job."""
        with _jobs_lock:
            if job_id not in _jobs:
                raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
            job = dict(_jobs[job_id])

        return {
            "job_id": job["job_id"],
            "status": job["status"],
            "stage": job["stage"],
            "progress": job["progress"],
            "created_at": job["created_at"],
            "updated_at": job["updated_at"],
            "error": job.get("error"),
        }

    @app.get("/jobs")
    def list_jobs() -> dict:
        """Lists all past and current jobs with their status."""
        with _jobs_lock:
            jobs_list = []
            for job_id, job in _jobs.items():
                jobs_list.append({
                    "job_id": job["job_id"],
                    "status": job["status"],
                    "stage": job["stage"],
                    "progress": job["progress"],
                    "model_id": job.get("model_id", ""),
                    "created_at": job["created_at"],
                })
        return {"jobs": jobs_list, "total": len(jobs_list)}

    @app.get("/results/{job_id}")
    def get_results(job_id: str) -> dict:
        """Returns paths to all output files for a completed job."""
        with _jobs_lock:
            if job_id not in _jobs:
                raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
            job = dict(_jobs[job_id])

        if job["status"] != "completed":
            raise HTTPException(
                status_code=400,
                detail=f"Job {job_id} is not completed (status: {job['status']})"
            )

        output_dir = job.get("output_dir", "crineforge_output")
        files = {}

        # Walk the output directory and list all files
        if os.path.exists(output_dir):
            for root, dirs, filenames in os.walk(output_dir):
                for filename in filenames:
                    rel_path = os.path.relpath(os.path.join(root, filename), output_dir)
                    files[rel_path] = os.path.join(root, filename)

        return {
            "job_id": job_id,
            "output_dir": output_dir,
            "files": files,
            "result": job.get("result"),
        }

    return app


# Create the app instance (lazy — only fails if FastAPI not installed when actually used)
try:
    app = _create_app()
except ImportError:
    app = None
    logger.debug("[API] FastAPI not available — API server disabled.")


def start_api(host: str = "0.0.0.0", port: int = 8000) -> None:
    """Starts the Crineforge API server using uvicorn."""
    if app is None:
        raise ImportError(
            "Cannot start API server.\n"
            "Reason: FastAPI and/or uvicorn are not installed.\n"
            "Suggested fix: Run `pip install fastapi uvicorn`"
        )

    try:
        import uvicorn
    except ImportError:
        raise ImportError(
            "uvicorn is not installed.\n"
            "Reason: Required to run the Crineforge API server.\n"
            "Suggested fix: Run `pip install uvicorn`"
        )

    logger.info(f"[API] Starting Crineforge API server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)
