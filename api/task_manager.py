import time
import uuid
import threading
from typing import Dict, Any, Optional

class TaskManager:
    """Thread-safe in-memory job manager for asynchronous signal processing."""
    
    def __init__(self, max_history: int = 100):
        self._jobs: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._max_history = max_history

    def create_job(self) -> str:
        job_id = str(uuid.uuid4())
        with self._lock:
            self._cleanup_old_jobs()
            self._jobs[job_id] = {
                "job_id": job_id,
                "status": "queued",
                "progress": 0.0,
                "stage": "Job queued for processing",
                "created_at": time.time(),
                "result": None,
                "error": None,
            }
        return job_id

    def update_job(
        self,
        job_id: str,
        status: Optional[str] = None,
        progress: Optional[float] = None,
        stage: Optional[str] = None,
        result: Optional[Any] = None,
        error: Optional[str] = None,
    ) -> None:
        with self._lock:
            if job_id not in self._jobs:
                return
            if status is not None:
                self._jobs[job_id]["status"] = status
            if progress is not None:
                self._jobs[job_id]["progress"] = float(progress)
            if stage is not None:
                self._jobs[job_id]["stage"] = str(stage)
            if result is not None:
                self._jobs[job_id]["result"] = result
            if error is not None:
                self._jobs[job_id]["error"] = str(error)

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is not None:
                return dict(job)
            return None

    def _cleanup_old_jobs(self, max_age_seconds: float = 3600.0) -> None:
        """Evict jobs older than 1 hour or when exceeding capacity."""
        now = time.time()
        to_delete = [
            jid for jid, j in self._jobs.items()
            if (now - j["created_at"]) > max_age_seconds
        ]
        for jid in to_delete:
            del self._jobs[jid]

        # If still over limit, evict oldest completed/failed jobs
        if len(self._jobs) > self._max_history:
            sorted_jobs = sorted(self._jobs.items(), key=lambda item: item[1]["created_at"])
            for jid, _ in sorted_jobs[:len(self._jobs) - self._max_history]:
                del self._jobs[jid]

# Global singleton
task_manager = TaskManager()
