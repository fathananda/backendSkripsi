"""
app/routers/speech.py
Router Speech-to-Text dan Speech-to-Translation.

Perbaikan:
- Whisper dan NLLB inference dijalankan via run_in_executor — tidak blocking.
"""

import asyncio
import logging
import time
from fastapi import APIRouter, Request, HTTPException, UploadFile, File, Form
from typing import Optional

from app.models.schemas import (
    SpeechToTextResponse,
    SpeechTranslateResponse,
    ErrorResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter()


def _get_whisper(request: Request):
    whisper = request.app.state.whisper_model
    if not whisper or not whisper.is_loaded:
        raise HTTPException(status_code=503, detail="Model Whisper belum siap.")
    return whisper


def _get_nllb(request: Request):
    nllb = request.app.state.nllb_model
    if not nllb or not nllb.is_loaded:
        raise HTTPException(status_code=503, detail="Model NLLB belum siap.")
    return nllb


@router.post(
    "/speech/transcribe",
    response_model=SpeechToTextResponse,
    responses={503: {"model": ErrorResponse}},
    summary="Transkripsi audio ke teks",
)
async def transcribe_audio(
    request: Request,
    audio: UploadFile = File(..., description="File audio (WAV/MP3/M4A)"),
    language: Optional[str] = Form("id", description="Kode bahasa audio (default: id)"),
):
    """Konversi file audio ke teks menggunakan Whisper."""
    whisper_model = _get_whisper(request)

    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="File audio kosong.")

    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(
            None,
            lambda: whisper_model.transcribe(audio_bytes, language=language),
        )
    except Exception as e:
        logger.error(f"Transcribe error: {e}")
        raise HTTPException(status_code=500, detail="Gagal melakukan transkripsi audio.")

    return {**result, "success": True}


@router.post(
    "/speech/translate",
    response_model=SpeechTranslateResponse,
    responses={503: {"model": ErrorResponse}},
    summary="Transkripsi audio lalu terjemahkan",
)
async def speech_to_translation(
    request: Request,
    audio: UploadFile = File(..., description="File audio (WAV/MP3/M4A)"),
    target_lang: str = Form(..., description="Kode bahasa target (jav, sun, ace, ...)"),
    source_lang: Optional[str] = Form("ind", description="Kode bahasa sumber (default: ind)"),
):
    """
    Pipeline lengkap: audio → Whisper (STT) → NLLB (translate).
    Kedua model dijalankan secara sequential di thread pool.
    """
    whisper_model = _get_whisper(request)
    nllb = _get_nllb(request)

    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="File audio kosong.")

    loop = asyncio.get_event_loop()
    total_start = time.time()

    # Step 1: STT dengan Whisper
    try:
        stt_result = await loop.run_in_executor(
            None,
            lambda: whisper_model.transcribe(audio_bytes, language="id"),
        )
    except Exception as e:
        logger.error(f"STT error: {e}")
        raise HTTPException(status_code=500, detail="Gagal melakukan transkripsi suara.")

    original_text = stt_result.get("text", "").strip()
    stt_time = stt_result.get("processing_time_ms", 0)

    if not original_text:
        raise HTTPException(status_code=422, detail="Tidak ada teks yang terdeteksi dari audio.")

    # Step 2: Translate dengan NLLB
    try:
        translate_result = await loop.run_in_executor(
            None,
            lambda: nllb.translate(
                text=original_text,
                source_lang=source_lang,
                target_lang=target_lang,
            ),
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Translate error in speech pipeline: {e}")
        raise HTTPException(status_code=500, detail="Gagal menerjemahkan teks.")

    translation_time = translate_result.get("processing_time_ms", 0)
    total_time = int((time.time() - total_start) * 1000)

    return {
        "success": True,
        "original_text": original_text,
        "translated_text": translate_result["translated_text"],
        "source_lang": source_lang,
        "target_lang": target_lang,
        "target_lang_name": translate_result["target_lang_name"],
        "stt_time_ms": stt_time,
        "translation_time_ms": translation_time,
        "total_time_ms": total_time,
    }
