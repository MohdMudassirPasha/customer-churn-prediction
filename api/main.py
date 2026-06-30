"""FastAPI application factory."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import router
from src.config import settings
from src.logger import get_logger, setup_logging
from src.predict import load_model_artifact

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Warm the model cache at startup so the first request avoids cold-load latency."""
    try:
        artifact = load_model_artifact(settings.model_path)
        logger.info("Model warmed at startup: %s", artifact.get("model_name", "unknown"))
    except Exception as exc:  # noqa: BLE001 - startup must not fail if the model is absent
        logger.warning("Model warm-up skipped (%s); it will load on first request.", exc)
    yield


def create_app() -> FastAPI:
    """Create and configure the FastAPI app."""
    setup_logging(level=settings.log_level, log_file=settings.log_path)
    app = FastAPI(
        title=settings.api_title,
        version=settings.api_version,
        description="Production API for IBM Telco customer churn prediction.",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(router)
    return app


app = create_app()
