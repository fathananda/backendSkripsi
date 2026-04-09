"""Router: Health Check"""
from fastapi import APIRouter, Request
from app.models.schemas import HealthResponse

router = APIRouter()

@router.get("/health", response_model=HealthResponse, summary="Cek status server")
async def health_check(request: Request):
    nllb = getattr(request.app.state, 'nllb_model', None)
    whisper = getattr(request.app.state, 'whisper_model', None)
    return HealthResponse(
        status="healthy" if (nllb and nllb.is_loaded and whisper and whisper.is_loaded) else "degraded",
        nllb_loaded=bool(nllb and nllb.is_loaded),
        whisper_loaded=bool(whisper and whisper.is_loaded),
        version="1.0.0",
    )
