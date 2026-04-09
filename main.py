"""
NusantaraTranslate Backend API
Aplikasi penerjemah bahasa daerah Indonesia menggunakan NLLB-200 dan Whisper
Author: Fathi Ananda Mas'ud
Prodi: Rekayasa Perangkat Lunak - Universitas BSI
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

from app.routers import translate, speech, language, health
from app.models.nllb_model import NLLBModel
from app.models.whisper_model import WhisperModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model instances
nllb_model: NLLBModel = None
whisper_model: WhisperModel = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup, cleanup on shutdown"""
    global nllb_model, whisper_model
    
    logger.info("🚀 Memuat model NLLB-200...")
    nllb_model = NLLBModel()
    nllb_model.load()
    app.state.nllb_model = nllb_model
    logger.info("✅ Model NLLB-200 berhasil dimuat")
    
    logger.info("🎙️ Memuat model Whisper...")
    whisper_model = WhisperModel()
    whisper_model.load()
    app.state.whisper_model = whisper_model
    logger.info("✅ Model Whisper berhasil dimuat")
    
    yield
    
    logger.info("🛑 Membersihkan resource...")


app = FastAPI(
    title="NusantaraTranslate API",
    description="""
    API penerjemah bahasa daerah Indonesia menggunakan Neural Machine Translation.
    
    ## Fitur
    - Terjemahan teks antara Bahasa Indonesia dan 7 bahasa daerah
    - Speech-to-Text untuk input suara Bahasa Indonesia
    - Evaluasi kualitas terjemahan
    
    ## Bahasa yang Didukung
    - 🇮🇩 Bahasa Indonesia
    - ☕ Bahasa Jawa
    - 🌺 Bahasa Sunda
    - 🌊 Bahasa Aceh
    - 🏝️ Bahasa Bali
    - ⚓ Bahasa Banjar
    - 🐝 Bahasa Bugis
    - 🏔️ Bahasa Minangkabau
    """,
    version="1.0.0",
    lifespan=lifespan
)

# CORS untuk akses dari Android
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


@app.get("/")
async def root():
    return {
        "app": "NusantaraTranslate API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs"
    }
