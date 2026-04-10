"""
app/models/whisper_model.py
Whisper Model Wrapper — Speech-to-Text untuk Bahasa Indonesia.
BLOCKING — caller (router) WAJIB memanggil via run_in_executor.
"""

import logging
import time
import tempfile
import os
import whisper

logger = logging.getLogger(__name__)

WHISPER_MODEL_SIZE = "small"


class WhisperModel:
    def __init__(self):
        self.model = None
        self.is_loaded = False

    def load(self):
        """Load model Whisper. Dipanggil sekali saat startup (lifespan)."""
        try:
            logger.info(f"Loading Whisper model: {WHISPER_MODEL_SIZE}")
            self.model = whisper.load_model(WHISPER_MODEL_SIZE)
            self.is_loaded = True
            logger.info("Whisper model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Whisper: {e}")
            raise

    def transcribe(self, audio_bytes: bytes, language: str = "id") -> dict:
        """
        Konversi audio ke teks. BLOCKING — panggil via run_in_executor di router.
        """
        if not self.is_loaded:
            raise RuntimeError("Model belum dimuat. Panggil load() terlebih dahulu.")

        start_time = time.time()

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            tmp_file.write(audio_bytes)
            tmp_path = tmp_file.name

        try:
            result = self.model.transcribe(
                tmp_path,
                language=language,
                task="transcribe",
                fp16=False,
                verbose=False,
            )
            processing_time = int((time.time() - start_time) * 1000)
            return {
                "text": result["text"].strip(),
                "language": result.get("language", language),
                "segments": [
                    {
                        "start": seg["start"],
                        "end": seg["end"],
                        "text": seg["text"].strip(),
                    }
                    for seg in result.get("segments", [])
                ],
                "processing_time_ms": processing_time,
                "model": f"whisper-{WHISPER_MODEL_SIZE}",
            }
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            raise
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def detect_language(self, audio_bytes: bytes) -> dict:
        """Deteksi bahasa dari audio. BLOCKING."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            tmp_file.write(audio_bytes)
            tmp_path = tmp_file.name
        try:
            audio = whisper.load_audio(tmp_path)
            audio = whisper.pad_or_trim(audio)
            mel = whisper.log_mel_spectrogram(audio).to(self.model.device)
            _, probs = self.model.detect_language(mel)
            detected = max(probs, key=probs.get)
            return {
                "detected_language": detected,
                "probability": float(probs[detected]),
            }
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
