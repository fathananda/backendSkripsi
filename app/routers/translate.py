"""
app/routers/translate.py
Router terjemahan teks.

Optimasi kecepatan:
- num_beams default = 1 (greedy) → ~4x lebih cepat
- Cache hasil translate selama 1 jam
- Non-blocking via run_in_executor
"""

import asyncio
import logging
from fastapi import APIRouter, Request, HTTPException, Query

from app.models.schemas import (
    TranslateRequest,
    TranslateResponse,
    BatchTranslateRequest,
    BatchTranslateResponse,
    ErrorResponse,
)
from app.utils.cache import translation_cache

logger = logging.getLogger(__name__)
router = APIRouter()


def _get_nllb(request: Request):
    nllb = request.app.state.nllb_model
    if not nllb or not nllb.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model NLLB belum siap. Coba beberapa saat lagi."
        )
    return nllb


@router.post(
    "/translate",
    response_model=TranslateResponse,
    responses={503: {"model": ErrorResponse}, 400: {"model": ErrorResponse}},
    summary="Terjemahkan teks",
)
async def translate_text(request: Request, body: TranslateRequest):
    """
    Terjemahkan teks dari bahasa sumber ke bahasa target.

    **Tips kecepatan:**
    - `num_beams=1` (default): greedy decoding, paling cepat (~1-3 detik di CPU)
    - `num_beams=4`: beam search, lebih akurat untuk teks panjang (~4-8 detik)
    """
    nllb = _get_nllb(request)

    # Cek cache terlebih dahulu — instant response jika sudah pernah diterjemahkan
    cached = translation_cache.get(
        body.text, body.source_lang, body.target_lang, body.num_beams
    )
    if cached:
        logger.info(f"Cache HIT: {body.source_lang}->{body.target_lang}")
        return {**cached, "processing_time_ms": 0}

    # Jalankan inference di thread pool — tidak blocking event loop FastAPI
    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(
            None,
            lambda: nllb.translate(
                text=body.text,
                source_lang=body.source_lang,
                target_lang=body.target_lang,
                num_beams=body.num_beams,
            ),
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Translate error: {e}")
        raise HTTPException(status_code=500, detail="Gagal menerjemahkan teks.")

    # Simpan ke cache
    translation_cache.set(
        body.text, body.source_lang, body.target_lang, result, body.num_beams
    )

    return result


@router.post(
    "/translate/batch",
    response_model=BatchTranslateResponse,
    responses={503: {"model": ErrorResponse}, 400: {"model": ErrorResponse}},
    summary="Terjemahkan banyak teks sekaligus",
)
async def batch_translate(request: Request, body: BatchTranslateRequest):
    """
    True batching — lebih efisien dari memanggil /translate berulang kali.
    """
    nllb = _get_nllb(request)

    if not body.texts:
        raise HTTPException(status_code=400, detail="Daftar teks tidak boleh kosong.")

    loop = asyncio.get_event_loop()
    try:
        results = await loop.run_in_executor(
            None,
            lambda: nllb.batch_translate(
                texts=body.texts,
                source_lang=body.source_lang,
                target_lang=body.target_lang,
                num_beams=body.num_beams,
            ),
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Batch translate error: {e}")
        raise HTTPException(status_code=500, detail="Gagal menerjemahkan batch teks.")

    return {"success": True, "total": len(results), "results": results}
