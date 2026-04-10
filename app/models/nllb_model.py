"""
app/models/nllb_model.py
NLLB-200 Model Wrapper — Neural Machine Translation.

OPTIMASI KECEPATAN (vs versi sebelumnya):
==========================================
1. num_beams default 1 (greedy) → ~4x lebih cepat dari beam=4
   Kualitas hampir sama untuk kalimat pendek-sedang.
   Beam=4 hanya berguna untuk teks sangat panjang/formal.

2. torch.set_num_threads() → pakai semua CPU core yang tersedia

3. Model di-compile dengan torch.compile() jika PyTorch >= 2.0
   → ~20-40% speedup pada CPU untuk inference berulang

4. int8 quantization opsional → ~2x lebih cepat, ~50% lebih hemat RAM
   (aktifkan dengan NLLB_QUANTIZE=1 di environment variable)

5. max_new_tokens menggantikan max_length → lebih presisi, tidak waste compute

6. Warmup saat startup → request pertama tidak lambat
"""

import logging
import os
import time
from typing import Optional
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

logger = logging.getLogger(__name__)

# ===== KONFIGURASI =====

# Pakai model distilled 600M — cukup bagus untuk bahasa daerah Indonesia
MODEL_NAME = "facebook/nllb-200-distilled-600M"

# Aktifkan quantization int8 untuk ~2x speedup di CPU
# Set env var: NLLB_QUANTIZE=1
USE_QUANTIZE = os.getenv("NLLB_QUANTIZE", "0") == "1"

# Default num_beams = 1 (greedy decoding) — jauh lebih cepat dari beam=4
# Untuk kalimat sehari-hari, kualitasnya hampir sama
DEFAULT_NUM_BEAMS = 1

LANGUAGE_CODES = {
    "ind": "ind_Latn",
    "jav": "jav_Latn",
    "sun": "sun_Latn",
    "ace": "ace_Latn",
    "ban": "ban_Latn",
    "bjn": "bjn_Latn",
    "bug": "bug_Latn",
    "min": "min_Latn",
}

LANGUAGE_NAMES = {
    "ind": "Bahasa Indonesia",
    "jav": "Bahasa Jawa",
    "sun": "Bahasa Sunda",
    "ace": "Bahasa Aceh",
    "ban": "Bahasa Bali",
    "bjn": "Bahasa Banjar",
    "bug": "Bahasa Bugis",
    "min": "Bahasa Minangkabau",
}


class NLLBModel:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.is_loaded = False

    def load(self):
        """
        Load dan optimalkan model NLLB-200.
        Dipanggil SEKALI saat startup.
        """
        try:
            logger.info(f"Loading NLLB-200 on device: {self.device}")

            # FIX 1: Pakai semua CPU core yang tersedia
            cpu_count = os.cpu_count() or 4
            torch.set_num_threads(cpu_count)
            torch.set_num_interop_threads(cpu_count)
            logger.info(f"PyTorch menggunakan {cpu_count} CPU threads")

            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

            if USE_QUANTIZE and self.device == "cpu":
                # FIX 2: INT8 Quantization — ~2x lebih cepat, ~50% hemat RAM
                # Aktifkan dengan: NLLB_QUANTIZE=1 uvicorn main:app ...
                logger.info("Loading NLLB-200 dengan INT8 quantization...")
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    MODEL_NAME,
                    quantization_config=quantization_config,
                    device_map="auto",
                )
            else:
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    MODEL_NAME,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                ).to(self.device)

            self.model.eval()

            # FIX 3: torch.compile — ~20-40% speedup (hanya PyTorch >= 2.0)
            # Kompilasi terjadi di background, tidak blocking startup
            torch_version = tuple(int(x) for x in torch.__version__.split(".")[:2])
            if torch_version >= (2, 0) and self.device == "cpu":
                try:
                    logger.info("Mengkompilasi model dengan torch.compile...")
                    self.model = torch.compile(self.model, mode="reduce-overhead")
                    logger.info("torch.compile berhasil")
                except Exception as e:
                    logger.warning(f"torch.compile gagal (tidak fatal): {e}")

            self.is_loaded = True
            logger.info("NLLB-200 berhasil dimuat")

            # FIX 4: Warmup — jalankan 1 translate dummy agar JIT/compile
            # selesai sebelum request pertama masuk
            self._warmup()

        except Exception as e:
            logger.error(f"Gagal load NLLB-200: {e}")
            raise

    def _warmup(self):
        """
        Warmup model dengan translate dummy.
        Mencegah request pertama user terasa lambat karena JIT compilation.
        """
        try:
            logger.info("Warming up model...")
            start = time.time()
            self.translate("Selamat pagi", "ind", "jav", num_beams=1)
            elapsed = int((time.time() - start) * 1000)
            logger.info(f"Warmup selesai dalam {elapsed}ms — siap menerima request")
        except Exception as e:
            logger.warning(f"Warmup gagal (tidak fatal): {e}")

    def translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        max_new_tokens: int = 256,
        num_beams: int = DEFAULT_NUM_BEAMS,
    ) -> dict:
        """
        Terjemahkan satu teks. BLOCKING — caller wajib pakai run_in_executor.

        num_beams=1 (default): greedy decoding, ~4x lebih cepat dari beam=4.
        num_beams=4: beam search, lebih akurat untuk teks panjang/formal.
        """
        if not self.is_loaded:
            raise RuntimeError("Model belum dimuat.")
        if source_lang not in LANGUAGE_CODES:
            raise ValueError(f"Bahasa sumber tidak didukung: {source_lang}")
        if target_lang not in LANGUAGE_CODES:
            raise ValueError(f"Bahasa target tidak didukung: {target_lang}")

        # Shortcircuit: tidak perlu translate jika bahasa sama
        if source_lang == target_lang:
            return {
                "translated_text": text,
                "source_lang": source_lang,
                "source_lang_name": LANGUAGE_NAMES[source_lang],
                "target_lang": target_lang,
                "target_lang_name": LANGUAGE_NAMES[target_lang],
                "processing_time_ms": 0,
                "model": MODEL_NAME,
                "char_count": len(text),
            }

        src_code = LANGUAGE_CODES[source_lang]
        tgt_code = LANGUAGE_CODES[target_lang]
        start_time = time.time()

        try:
            self.tokenizer.src_lang = src_code
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(self.device)

            with torch.no_grad():
                translated_tokens = self.model.generate(
                    **inputs,
                    forced_bos_token_id=self.tokenizer.convert_tokens_to_ids(tgt_code),
                    # FIX 5: max_new_tokens lebih efisien dari max_length
                    # max_length menghitung input+output, max_new_tokens hanya output
                    max_new_tokens=max_new_tokens,
                    num_beams=num_beams,
                    # FIX 6: early_stopping hanya berguna saat num_beams > 1
                    early_stopping=(num_beams > 1),
                    # FIX 7: Kurangi no_repeat_ngram dari 3 ke 2
                    # ngram=3 menyebabkan lebih banyak candidate yang dievaluasi
                    no_repeat_ngram_size=2 if num_beams > 1 else 0,
                )

            translated_text = self.tokenizer.batch_decode(
                translated_tokens, skip_special_tokens=True
            )[0]

            processing_time = int((time.time() - start_time) * 1000)
            logger.info(
                f"Translated [{source_lang}→{target_lang}] "
                f"{len(text)} chars in {processing_time}ms "
                f"(beams={num_beams})"
            )

            return {
                "translated_text": translated_text,
                "source_lang": source_lang,
                "source_lang_name": LANGUAGE_NAMES[source_lang],
                "target_lang": target_lang,
                "target_lang_name": LANGUAGE_NAMES[target_lang],
                "processing_time_ms": processing_time,
                "model": MODEL_NAME,
                "char_count": len(text),
            }

        except Exception as e:
            logger.error(f"Translation error: {e}")
            raise

    def batch_translate(
        self,
        texts: list,
        source_lang: str,
        target_lang: str,
        num_beams: int = DEFAULT_NUM_BEAMS,
    ) -> list:
        """
        True batching — semua teks diproses sekaligus.
        BLOCKING — caller wajib pakai run_in_executor.
        """
        if not self.is_loaded:
            raise RuntimeError("Model belum dimuat.")
        if source_lang not in LANGUAGE_CODES:
            raise ValueError(f"Bahasa sumber tidak didukung: {source_lang}")
        if target_lang not in LANGUAGE_CODES:
            raise ValueError(f"Bahasa target tidak didukung: {target_lang}")
        if not texts:
            return []

        src_code = LANGUAGE_CODES[source_lang]
        tgt_code = LANGUAGE_CODES[target_lang]
        start_time = time.time()

        try:
            self.tokenizer.src_lang = src_code
            inputs = self.tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(self.device)

            with torch.no_grad():
                translated_tokens = self.model.generate(
                    **inputs,
                    forced_bos_token_id=self.tokenizer.convert_tokens_to_ids(tgt_code),
                    max_new_tokens=256,
                    num_beams=num_beams,
                    early_stopping=(num_beams > 1),
                    no_repeat_ngram_size=2 if num_beams > 1 else 0,
                )

            translated_texts = self.tokenizer.batch_decode(
                translated_tokens, skip_special_tokens=True
            )

            total_time = int((time.time() - start_time) * 1000)
            avg_time = total_time // max(len(texts), 1)

            return [
                {
                    "translated_text": t_text,
                    "source_lang": source_lang,
                    "source_lang_name": LANGUAGE_NAMES[source_lang],
                    "target_lang": target_lang,
                    "target_lang_name": LANGUAGE_NAMES[target_lang],
                    "processing_time_ms": avg_time,
                    "model": MODEL_NAME,
                    "char_count": len(orig),
                }
                for t_text, orig in zip(translated_texts, texts)
            ]

        except Exception as e:
            logger.error(f"Batch translation error: {e}")
            raise

    def get_supported_languages(self) -> dict:
        return {
            code: {"name": LANGUAGE_NAMES[code], "nllb_code": LANGUAGE_CODES[code]}
            for code in LANGUAGE_CODES
        }
