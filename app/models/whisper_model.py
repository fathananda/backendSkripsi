"""
Whisper Model Wrapper
Model Speech-to-Text dari OpenAI untuk konversi suara ke teks Bahasa Indonesia
"""

import logging
import time
import tempfile
import os
from typing import Optional
import whisper
import numpy as np

logger = logging.getLogger(__name__)

# Gunakan model 'small' untuk keseimbangan akurasi dan kecepatan
WHISPER_MODEL_SIZE = "small"


class WhisperModel:
    def __init__(self):
        self.model = None
        self.is_loaded = False

    def load(self):
        """Load model Whisper"""
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
        Konversi audio ke teks menggunakan Whisper
        
        Args:
            audio_bytes: File audio dalam bentuk bytes (WAV/MP3/M4A)
            language: Kode bahasa audio (default: 'id' untuk Indonesia)
            
        Returns:
            dict berisi teks hasil transkripsi dan metadata
        """
        if not self.is_loaded:
            raise RuntimeError("Model belum dimuat. Panggil load() terlebih dahulu.")

        start_time = time.time()

        # Simpan audio ke file temporary
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            tmp_file.write(audio_bytes)
            tmp_path = tmp_file.name

        try:
            # Transkripsi dengan Whisper
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
            # Hapus file temporary
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def detect_language(self, audio_bytes: bytes) -> dict:
        """Deteksi bahasa dari audio"""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            tmp_file.write(audio_bytes)
            tmp_path = tmp_file.name
        try:
            audio = whisper.load_audio(tmp_path)
            audio = whisper.pad_or_trim(audio)
            mel = whisper.log_mel_spectrogram(audio).to(self.model.device)
            _, probs = self.model.detect_language(mel)
            detected = max(probs, key=probs.get)
            return {"detected_language": detected, "probability": float(probs[detected])}
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
