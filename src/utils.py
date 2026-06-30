"""Shared utilities for data, artifacts, and reproducibility."""

from __future__ import annotations

import json
import random
import urllib.request
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.config import Settings, settings
from src.logger import get_logger

logger = get_logger(__name__)


def set_global_seed(seed: int) -> None:
    """Set deterministic seeds for supported libraries."""
    random.seed(seed)
    np.random.seed(seed)


def ensure_project_directories(config: Settings = settings) -> None:
    """Ensure all runtime directories exist."""
    config.ensure_directories()


def download_dataset(
    destination: str | Path | None = None,
    url: str | None = None,
    overwrite: bool = False,
    timeout: int = 30,
    config: Settings = settings,
) -> Path:
    """Download the IBM Telco dataset to the raw data directory."""
    ensure_project_directories(config)
    path = Path(destination) if destination is not None else config.raw_data_path
    dataset_url = url or config.data_url

    if path.exists() and not overwrite:
        logger.info("Dataset already exists at %s", path)
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading dataset from %s", dataset_url)
    with urllib.request.urlopen(dataset_url, timeout=timeout) as response:
        content = response.read()
    path.write_bytes(content)
    logger.info("Saved dataset to %s", path)
    return path


def load_dataset(
    path: str | Path | None = None,
    allow_download: bool = False,
    config: Settings = settings,
) -> pd.DataFrame:
    """Load the Telco churn dataset from disk, optionally downloading it."""
    dataset_path = Path(path) if path is not None else config.raw_data_path
    if not dataset_path.exists():
        if path is None and config.sample_data_path.exists():
            logger.warning(
                "Dataset not found at %s; using bundled sample dataset at %s.",
                dataset_path,
                config.sample_data_path,
            )
            dataset_path = config.sample_data_path
        elif allow_download:
            dataset_path = download_dataset(destination=dataset_path, config=config)
        else:
            raise FileNotFoundError(
                f"Dataset not found at {dataset_path}. "
                "Place the IBM Telco Customer Churn CSV there, provide a sample dataset, "
                "or run with allow_download=True."
            )

    logger.info("Loading dataset from %s", dataset_path)
    return pd.read_csv(dataset_path)


def load_sample_dataset(config: Settings = settings) -> pd.DataFrame:
    """Load the bundled sample Telco churn dataset."""
    if not config.sample_data_path.exists():
        raise FileNotFoundError(f"Sample dataset not found at {config.sample_data_path}")
    logger.info("Loading sample dataset from %s", config.sample_data_path)
    return pd.read_csv(config.sample_data_path)


def save_dataframe(data: pd.DataFrame, path: str | Path) -> Path:
    """Persist a dataframe as CSV."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_path, index=False)
    return output_path


def save_json(payload: dict[str, Any], path: str | Path) -> Path:
    """Persist JSON with deterministic formatting."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True, default=str)
    return output_path


def load_json(path: str | Path) -> dict[str, Any]:
    """Load a JSON file."""
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def save_joblib(obj: Any, path: str | Path) -> Path:
    """Persist an object with joblib."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, output_path)
    return output_path


def load_joblib(path: str | Path) -> Any:
    """Load a joblib artifact."""
    artifact_path = Path(path)
    if not artifact_path.exists():
        raise FileNotFoundError(f"Artifact not found at {artifact_path}")
    return joblib.load(artifact_path)


def dataframe_to_records(data: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert a dataframe to JSON-safe records."""
    return json.loads(data.replace({np.nan: None}).to_json(orient="records"))


def build_model_card(
    model_name: str,
    metrics: dict[str, float],
    features: list[str],
    artifact_path: Path,
    threshold: float,
) -> dict[str, Any]:
    """Create model metadata for API and audit usage."""
    return {
        "model_name": model_name,
        "artifact_path": str(artifact_path),
        "decision_threshold": threshold,
        "metrics": metrics,
        "feature_count": len(features),
        "features": features,
    }
