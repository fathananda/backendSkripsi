"""
NLLB-200 Model Wrapper
Model Neural Machine Translation dari Meta AI untuk 7 bahasa daerah Indonesia
"""

import logging
import time
from typing import Optional
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

logger = logging.getLogger(__name__)

# Kode bahasa NLLB-200 untuk bahasa daerah Indonesia
LANGUAGE_CODES = {
    "ind": "ind_Latn",   # Bahasa Indonesia
    "jav": "jav_Latn",   # Bahasa Jawa
    "sun": "sun_Latn",   # Bahasa Sunda
    "ace": "ace_Latn",   # Bahasa Aceh
    "ban": "ban_Latn",   # Bahasa Bali
    "bjn": "bjn_Latn",   # Bahasa Banjar
    "bug": "bug_Latn",   # Bahasa Bugis
    "min": "min_Latn",   # Bahasa Minangkabau
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

# Gunakan model distilled 600M agar lebih ringan
MODEL_NAME = "facebook/nllb-200-distilled-600M"


class NLLBModel:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.is_loaded = False

    def load(self):
        """Load model NLLB-200 dari HuggingFace"""
        try:
            logger.info(f"Loading NLLB-200 on device: {self.device}")
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                MODEL_NAME,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            ).to(self.device)
            self.model.eval()
            self.is_loaded = True
            logger.info("NLLB-200 loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load NLLB-200: {e}")
            raise

    def translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        max_length: int = 512,
        num_beams: int = 4,
    ) -> dict:
        """
        Terjemahkan teks menggunakan NLLB-200
        
        Args:
            text: Teks yang akan diterjemahkan
            source_lang: Kode bahasa sumber (misal: 'ind')
            target_lang: Kode bahasa target (misal: 'jav')
            max_length: Panjang maksimum output
            num_beams: Jumlah beam untuk beam search
            
        Returns:
            dict berisi hasil terjemahan dan metadata
        """
        if not self.is_loaded:
            raise RuntimeError("Model belum dimuat. Panggil load() terlebih dahulu.")

        if source_lang not in LANGUAGE_CODES:
            raise ValueError(f"Bahasa sumber tidak didukung: {source_lang}")
        if target_lang not in LANGUAGE_CODES:
            raise ValueError(f"Bahasa target tidak didukung: {target_lang}")
        if source_lang == target_lang:
            return {
                "translated_text": text,
                "source_lang": source_lang,
                "target_lang": target_lang,
                "processing_time_ms": 0,
                "model": MODEL_NAME,
            }

        src_code = LANGUAGE_CODES[source_lang]
        tgt_code = LANGUAGE_CODES[target_lang]

        start_time = time.time()

        try:
            self.tokenizer.src_lang = src_code

            # Tokenisasi input
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(self.device)

            # Generate terjemahan
            with torch.no_grad():
                translated_tokens = self.model.generate(
                    **inputs,
                    forced_bos_token_id=self.tokenizer.convert_tokens_to_ids(tgt_code),
                    max_length=max_length,
                    num_beams=num_beams,
                    early_stopping=True,
                    no_repeat_ngram_size=3,
                )

            # Decode hasil
            translated_text = self.tokenizer.batch_decode(
                translated_tokens, skip_special_tokens=True
            )[0]

            processing_time = int((time.time() - start_time) * 1000)

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

    def batch_translate(self, texts: list, source_lang: str, target_lang: str) -> list:
        """Terjemahkan banyak teks sekaligus (untuk evaluasi BLEU Score)"""
        results = []
        for text in texts:
            result = self.translate(text, source_lang, target_lang)
            results.append(result)
        return results

    def get_supported_languages(self) -> dict:
        """Dapatkan daftar bahasa yang didukung"""
        return {
            code: {
                "name": LANGUAGE_NAMES[code],
                "nllb_code": LANGUAGE_CODES[code],
            }
            for code in LANGUAGE_CODES
        }
