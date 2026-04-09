"""
Router: Speech
Endpoint Speech-to-Text menggunakan Whisper + terjemahan otomatis
"""

from fastapi import APIRouter, Request, HTTPException, UploadFile, File, Form
from app.models.schemas import SpeechToTextResponse, SpeechTranslateResponse
import logging
import time

logger = logging.getLogger(__name__)
router = APIRouter()

ALLOWED_AUDIO_TYPES = {
    "audio/wav", "audio/wave", "audio/x-wav",
    "audio/mpeg", "audio/mp3",
    "audio/mp4", "audio/m4a",
    "audio/ogg", "audio/webm",
    "application/octet-stream",
}
MAX_AUDIO_SIZE = 25 * 1024 * 1024  # 25MB


@router.post(
    "/speech/transcribe",
    response_model=SpeechToTextResponse,
    summary="Konversi suara ke teks",
    description="Transkripsi audio Bahasa Indonesia menggunakan Whisper"
)
async def transcribe_audio(
    request: Request,
    audio: UploadFile = File(..., description="File audio (WAV/MP3/M4A, max 25MB)"),
    language: str = Form(default="id", description="Kode bahasa audio (default: id)")
):
    """
    Konversi file audio ke teks menggunakan model Whisper.
    Mendukung format WAV, MP3, M4A, OGG, WebM.
    """
    whisper_model = request.app.state.whisper_model
    if not whisper_model or not whisper_model.is_loaded:
        raise HTTPException(status_code=503, detail="Model Whisper belum siap")

    # Validasi ukuran file
    audio_bytes = await audio.read()
    if len(audio_bytes) > MAX_AUDIO_SIZE:
        raise HTTPException(status_code=413, detail="Ukuran file audio melebihi 25MB")
    if len(audio_bytes) == 0:
        raise HTTPException(status_code=400, detail="File audio kosong")

    try:
        result = whisper_model.transcribe(audio_bytes, language=language)
        return SpeechToTextResponse(**result, success=True)
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        raise HTTPException(status_code=500, detail="Terjadi kesalahan saat transkripsi")


@router.post(
    "/speech/translate",
    response_model=SpeechTranslateResponse,
    summary="Suara langsung ke terjemahan",
    description="Pipeline lengkap: Audio → Teks (Whisper) → Terjemahan (NLLB-200)"
)
async def speech_to_translation(
    request: Request,
    audio: UploadFile = File(..., description="File audio Bahasa Indonesia"),
    target_lang: str = Form(..., description="Kode bahasa target terjemahan"),
    source_lang: str = Form(default="ind", description="Kode bahasa sumber (default: ind)"),
):
    """
    Pipeline lengkap: Suara → Teks → Terjemahan.
    
    Novelty: Menggabungkan Whisper (STT) dan NLLB-200 (NMT)
    dalam satu endpoint untuk pengalaman pengguna yang seamless.
    """
    whisper_model = request.app.state.whisper_model
    nllb_model = request.app.state.nllb_model

    if not whisper_model or not whisper_model.is_loaded:
        raise HTTPException(status_code=503, detail="Model Whisper belum siap")
    if not nllb_model or not nllb_model.is_loaded:
        raise HTTPException(status_code=503, detail="Model NLLB-200 belum siap")

    audio_bytes = await audio.read()
    if len(audio_bytes) == 0:
        raise HTTPException(status_code=400, detail="File audio kosong")

    total_start = time.time()

    try:
        # Step 1: Speech-to-Text dengan Whisper
        stt_result = whisper_model.transcribe(audio_bytes, language="id")
        original_text = stt_result["text"]

        if not original_text.strip():
            raise HTTPException(status_code=422, detail="Tidak ada teks yang terdeteksi dari audio")

        # Step 2: Terjemahan dengan NLLB-200
        translation_result = nllb_model.translate(
            text=original_text,
            source_lang=source_lang,
            target_lang=target_lang,
        )

        total_time = int((time.time() - total_start) * 1000)

        return SpeechTranslateResponse(
            success=True,
            original_text=original_text,
            translated_text=translation_result["translated_text"],
            source_lang=source_lang,
            target_lang=target_lang,
            target_lang_name=translation_result["target_lang_name"],
            stt_time_ms=stt_result["processing_time_ms"],
            translation_time_ms=translation_result["processing_time_ms"],
            total_time_ms=total_time,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Speech translate pipeline error: {e}")
        raise HTTPException(status_code=500, detail="Terjadi kesalahan pada pipeline suara")
