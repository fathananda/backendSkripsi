"""
Router: Translation
Endpoint untuk terjemahan teks menggunakan NLLB-200
"""

from fastapi import APIRouter, Request, HTTPException
from app.models.schemas import (
    TranslateRequest, TranslateResponse,
    BatchTranslateRequest, ErrorResponse
)
import logging

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post(
    "/translate",
    response_model=TranslateResponse,
    summary="Terjemahkan teks",
    description="Terjemahkan teks antara Bahasa Indonesia dan bahasa daerah menggunakan NLLB-200"
)
async def translate_text(request: Request, body: TranslateRequest):
    """
    Endpoint utama terjemahan teks.
    
    - **text**: Teks yang akan diterjemahkan (max 2000 karakter)
    - **source_lang**: Kode bahasa sumber (ind/jav/sun/ace/ban/bjn/bug/min)
    - **target_lang**: Kode bahasa target
    - **num_beams**: Kualitas terjemahan (1=cepat, 8=akurat)
    """
    nllb_model = request.app.state.nllb_model
    if not nllb_model or not nllb_model.is_loaded:
        raise HTTPException(status_code=503, detail="Model NLLB-200 belum siap")

    try:
        result = nllb_model.translate(
            text=body.text,
            source_lang=body.source_lang,
            target_lang=body.target_lang,
            num_beams=body.num_beams,
        )
        return TranslateResponse(**result, success=True)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Translation error: {e}")
        raise HTTPException(status_code=500, detail="Terjadi kesalahan saat menerjemahkan")


@router.post(
    "/translate/batch",
    summary="Terjemahkan banyak teks sekaligus",
    description="Terjemahkan multiple teks — berguna untuk evaluasi BLEU Score"
)
async def batch_translate(request: Request, body: BatchTranslateRequest):
    """
    Terjemahkan banyak kalimat sekaligus.
    Berguna untuk evaluasi performa model menggunakan BLEU Score.
    """
    nllb_model = request.app.state.nllb_model
    if not nllb_model or not nllb_model.is_loaded:
        raise HTTPException(status_code=503, detail="Model NLLB-200 belum siap")

    try:
        results = nllb_model.batch_translate(
            texts=body.texts,
            source_lang=body.source_lang,
            target_lang=body.target_lang,
        )
        return {
            "success": True,
            "total": len(results),
            "results": results,
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Batch translation error: {e}")
        raise HTTPException(status_code=500, detail="Terjadi kesalahan saat batch translate")
