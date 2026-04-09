"""
Pydantic Schemas — definisi request dan response API
"""

from pydantic import BaseModel, Field
from typing import Optional, List


# ===== REQUEST SCHEMAS =====

class TranslateRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000, description="Teks yang akan diterjemahkan")
    source_lang: str = Field(..., description="Kode bahasa sumber (misal: ind, jav, sun)")
    target_lang: str = Field(..., description="Kode bahasa target (misal: jav, ind, sun)")
    num_beams: Optional[int] = Field(4, ge=1, le=8, description="Jumlah beam untuk beam search")

    class Config:
        json_schema_extra = {
            "example": {
                "text": "Saya ingin pergi ke pasar membeli sayuran",
                "source_lang": "ind",
                "target_lang": "jav",
                "num_beams": 4
            }
        }


class BatchTranslateRequest(BaseModel):
    texts: List[str] = Field(..., min_length=1, max_length=100, description="Daftar teks")
    source_lang: str
    target_lang: str

    class Config:
        json_schema_extra = {
            "example": {
                "texts": ["Selamat pagi", "Apa kabar?", "Terima kasih"],
                "source_lang": "ind",
                "target_lang": "sun"
            }
        }


# ===== RESPONSE SCHEMAS =====

class TranslateResponse(BaseModel):
    success: bool = True
    translated_text: str
    source_lang: str
    source_lang_name: str
    target_lang: str
    target_lang_name: str
    processing_time_ms: int
    model: str
    char_count: int


class SpeechToTextResponse(BaseModel):
    success: bool = True
    text: str
    language: str
    segments: List[dict]
    processing_time_ms: int
    model: str


class SpeechTranslateResponse(BaseModel):
    success: bool = True
    original_text: str
    translated_text: str
    source_lang: str
    target_lang: str
    target_lang_name: str
    stt_time_ms: int
    translation_time_ms: int
    total_time_ms: int


class LanguageInfo(BaseModel):
    code: str
    name: str
    nllb_code: str
    flag: Optional[str] = None
    speakers_million: Optional[float] = None
    region: Optional[str] = None


class LanguageListResponse(BaseModel):
    success: bool = True
    total: int
    languages: List[LanguageInfo]


class HealthResponse(BaseModel):
    status: str
    nllb_loaded: bool
    whisper_loaded: bool
    version: str


class ErrorResponse(BaseModel):
    success: bool = False
    error: str
    detail: Optional[str] = None
