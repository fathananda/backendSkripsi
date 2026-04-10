"""
NusantaraTranslate Backend API
Penerjemah bahasa daerah Indonesia menggunakan NLLB-200 dan Whisper.

Perubahan dari versi sebelumnya:
- Semua inference dijalankan non-blocking via asyncio.run_in_executor()
- Cache hasil terjemahan untuk mengurangi beban model
- True batching di batch_translate (semua teks sekaligus, bukan serial loop)
- Logging lebih informatif
- Health check menyertakan statistik cache
"""

import logging
import warnings
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Suppress PyTorch TypedStorage deprecation warning — tidak berbahaya,
# hanya noise dari internal PyTorch saat load model transformers.
warnings.filterwarnings(
    "ignore",
    message="TypedStorage is deprecated",
    category=UserWarning,
    module="torch",
)

from app.routers import translate, speech, language, health
from app.models.nllb_model import NLLBModel
from app.models.whisper_model import WhisperModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load model saat startup, bersihkan resource saat shutdown.
    Model di-load SEKALI sebagai singleton — tidak per-request.
    """
    logger.info("🚀 Memuat model NLLB-200...")
    nllb = NLLBModel()
    nllb.load()
    app.state.nllb_model = nllb
    logger.info("✅ NLLB-200 berhasil dimuat")

    logger.info("🎙️ Memuat model Whisper...")
    whisper = WhisperModel()
    whisper.load()
    app.state.whisper_model = whisper
    logger.info("✅ Whisper berhasil dimuat")

    logger.info("🟢 Server siap menerima request")
    yield

    logger.info("🛑 Membersihkan resource...")
    # Bebaskan VRAM/RAM jika perlu
    app.state.nllb_model = None
    app.state.whisper_model = None


app = FastAPI(
    title="NusantaraTranslate API",
    description="""
API penerjemah bahasa daerah Indonesia menggunakan Neural Machine Translation.

## Bahasa yang Didukung
🇮🇩 Indonesia · ☕ Jawa · 🌺 Sunda · 🌊 Aceh · 🏝️ Bali · ⚓ Banjar · 🐝 Bugis · 🏔️ Minangkabau

## Catatan Performa
- Request pertama lebih lambat (model warm-up)
- Hasil terjemahan di-cache 1 jam untuk teks yang sama
- Gunakan `/translate/batch` untuk menerjemahkan banyak teks sekaligus
    """,
    version="1.1.0",
    lifespan=lifespan,
)

# CORS — izinkan akses dari Android (dan semua origin untuk development)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(health.router, prefix="/api/v1", tags=["Health"])
app.include_router(language.router, prefix="/api/v1", tags=["Languages"])
app.include_router(translate.router, prefix="/api/v1", tags=["Translation"])
app.include_router(speech.router, prefix="/api/v1", tags=["Speech"])


@app.get("/", tags=["Root"])
async def root():
    return {
        "app": "NusantaraTranslate API",
        "version": "1.1.0",
        "status": "running",
        "docs": "/docs",
        "endpoints": {
            "health": "/api/v1/health",
            "languages": "/api/v1/languages",
            "translate": "/api/v1/translate",
            "batch_translate": "/api/v1/translate/batch",
            "transcribe": "/api/v1/speech/transcribe",
            "speech_translate": "/api/v1/speech/translate",
        },
    }