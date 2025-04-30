import time
from datetime import datetime
import numpy as np
import json
from pathlib import Path
import psutil
from typing import Dict, List, Optional

class RAGMetrics:
    def __init__(self, log_dir: str = "metrics_logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.query_times = []
        self.retrieval_times = []
        self.similarity_scores = []
        self.error_log = self.log_dir / "error_log.jsonl"
        self.query_log = self.log_dir / "query_log.jsonl"

    def log_short_query(self, query: str) -> None:
        with open(self.query_log, "a") as f:
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "query": query,
                "type": "short_query",
                "length": len(query)
            }
            f.write(json.dumps(log_entry) + "\n")

    def log_empty_results(self, query: str) -> None:
        with open(self.query_log, "a") as f:
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "query": query,
                "type": "no_results"
            }
            f.write(json.dumps(log_entry) + "\n")

    def log_error(self, query: str, error: str, duration: float) -> None:
        with open(self.error_log, "a") as f:
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "query": query,
                "error": error,
                "duration": duration
            }
            f.write(json.dumps(log_entry) + "\n")

    def log_query_metrics(self, query: str, results: List[Dict], 
                         start_time: float, search_time: float, 
                         processing_time: float) -> Dict:
        duration = time.time() - start_time
        metrics = {
            "query_time": duration,
            "search_time": search_time,
            "processing_time": processing_time,
            "num_results": len(results),
            "avg_similarity": np.mean([r["similarity"] for r in results]) if results else 0
        }
        with open(self.query_log, "a") as f:
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "query": query,
                "metrics": metrics
            }
            f.write(json.dumps(log_entry) + "\n")      
        return metrics

    def get_system_stats(self) -> Dict:
        return {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "queries_processed": len(self.query_times),
            "avg_query_time": np.mean(self.query_times) if self.query_times else 0,
            "total_errors": sum(1 for _ in open(self.error_log)) if self.error_log.exists() else 0
        }