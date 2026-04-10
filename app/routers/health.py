"""
app/routers/health.py
Health check endpoint — menampilkan status model dan cache.
"""

import logging
from fastapi import APIRouter, Request
from app.models.schemas import HealthResponse
from app.utils.cache import translation_cache

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Status kesehatan server",
)
async def health_check(request: Request):
    """Cek apakah server dan model berjalan dengan baik."""
    nllb = getattr(request.app.state, "nllb_model", None)
    whisper = getattr(request.app.state, "whisper_model", None)

    return HealthResponse(
        status="ok",
        nllb_loaded=nllb is not None and nllb.is_loaded,
        whisper_loaded=whisper is not None and whisper.is_loaded,
        version="1.1.0",
        cache_stats=translation_cache.stats(),
    )
