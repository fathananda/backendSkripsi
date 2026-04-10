"""
app/routers/language.py
Router daftar bahasa yang didukung.
"""

import logging
from fastapi import APIRouter
from app.models.schemas import LanguageListResponse, LanguageInfo

logger = logging.getLogger(__name__)
router = APIRouter()

LANGUAGES = [
    LanguageInfo(code="ind", name="Bahasa Indonesia", nllb_code="ind_Latn",
                 flag="🇮🇩", speakers_million=270.0, region="Nasional"),
    LanguageInfo(code="jav", name="Bahasa Jawa", nllb_code="jav_Latn",
                 flag="☕", speakers_million=98.0, region="Jawa Tengah, Jawa Timur"),
    LanguageInfo(code="sun", name="Bahasa Sunda", nllb_code="sun_Latn",
                 flag="🌺", speakers_million=42.0, region="Jawa Barat"),
    LanguageInfo(code="ace", name="Bahasa Aceh", nllb_code="ace_Latn",
                 flag="🌊", speakers_million=3.5, region="Aceh"),
    LanguageInfo(code="ban", name="Bahasa Bali", nllb_code="ban_Latn",
                 flag="🏝️", speakers_million=3.3, region="Bali"),
    LanguageInfo(code="bjn", name="Bahasa Banjar", nllb_code="bjn_Latn",
                 flag="⚓", speakers_million=3.5, region="Kalimantan Selatan"),
    LanguageInfo(code="bug", name="Bahasa Bugis", nllb_code="bug_Latn",
                 flag="🐝", speakers_million=5.0, region="Sulawesi Selatan"),
    LanguageInfo(code="min", name="Bahasa Minangkabau", nllb_code="min_Latn",
                 flag="🏔️", speakers_million=6.5, region="Sumatera Barat"),
]


@router.get(
    "/languages",
    response_model=LanguageListResponse,
    summary="Daftar bahasa yang didukung",
)
async def get_languages():
    """Kembalikan daftar lengkap bahasa daerah yang didukung beserta metadata."""
    return LanguageListResponse(
        success=True,
        total=len(LANGUAGES),
        languages=LANGUAGES,
    )
