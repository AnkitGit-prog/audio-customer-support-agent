"""
FastAPI server for the Audio Customer Support Agent.

Exposes the full pipeline via REST API endpoints:
  GET  /health          - component health status
  POST /chat/text       - text in, text out
  POST /chat/audio      - audio in, audio out (full pipeline)
  GET  /chat/audio/{text} - TTS only (quick test)
  POST /debug/stt       - STT only (transcription test)

Run with:
  python -m src.api.server
  # or
  uvicorn src.api.server:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import io
import logging
import asyncio
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Dict, Any

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

from src.pipeline import AudioPipeline  # noqa: E402

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Audio Customer Support Agent",
    description=(
        "Voice-based customer support bot. "
        "Pipeline: Audio → Whisper STT → GPT-3.5 + ChromaDB RAG → Edge TTS → Audio"
    ),
    version="1.0.0",
)

# CORS — allow all origins so Streamlit frontend can connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Global pipeline instance
# ---------------------------------------------------------------------------
pipeline = None

@app.on_event("startup")
async def startup_event():
    global pipeline
    logger.info("Starting server and initializing pipeline...")
    
    # Configure services according to PDF
    stt_config = {
        "model": os.getenv("STT_MODEL", "base")
    }
    llm_config = {
        "openai_api_key": os.getenv("OPENAI_API_KEY"),
        "llm_model": os.getenv("LLM_MODEL", "gpt-3.5-turbo"),
        "llm_temperature": float(os.getenv("LLM_TEMPERATURE", "0.7")),
        "llm_base_url": os.getenv("LLM_BASE_URL")
    }
    tts_config = {
        "voice": os.getenv("TTS_VOICE", "en-US-AriaNeural")
    }
    
    pipeline = AudioPipeline(stt_config, llm_config, tts_config)
    await pipeline.initialize()
    logger.info("Pipeline initialized and ready.")


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------
class TextRequest(BaseModel):
    """Request body for text-based chat."""
    text: str


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def _require_pipeline() -> None:
    """Raise HTTP 503 if the pipeline is not initialised."""
    if pipeline is None or not pipeline.is_initialized:
        raise HTTPException(
            status_code=503,
            detail="Pipeline not initialised. Check server logs for details.",
        )

@app.get("/health")
async def health():
    if pipeline is None or not pipeline.is_initialized:
        return JSONResponse(
            status_code=503,
            content={"status": "not_ready", "components": pipeline.get_status() if pipeline else {}}
        )
    return {"status": "ready", "components": pipeline.get_status()}

@app.post("/chat/text")
async def chat_text(request: TextRequest, session_id: str = "default"):
    _require_pipeline()
    response = await pipeline.process_text(request.text, session_id=session_id)
    return {"response": response, "session_id": session_id}

@app.post("/chat/audio")
async def chat_audio(
    audio: UploadFile = File(...), 
    session_id: str = "default",
    stream: bool = False
):
    _require_pipeline()
    audio_bytes = await audio.read()
    
    # Process through pipeline
    result = await pipeline.process_audio(audio_bytes, session_id=session_id, stream=stream)
    
    headers = {
        "X-Transcript": result["transcript"],
        "X-Response": result["response"],
        "X-Language": result["language"],
        "X-Session-ID": session_id,
        "Content-Disposition": "attachment; filename=response.mp3"
    }

    if stream:
        return StreamingResponse(
            result["audio"], 
            media_type="audio/mpeg", 
            headers=headers
        )
    else:
        return StreamingResponse(
            io.BytesIO(result["audio"]),
            media_type="audio/mpeg",
            headers=headers
        )

@app.get("/chat/audio/{text}")
async def chat_audio_text(text: str, language: str = "en") -> StreamingResponse:
    _require_pipeline()
    if not text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
    audio_bytes: bytes = await pipeline.text_to_audio(text)
    return StreamingResponse(
        io.BytesIO(audio_bytes),
        media_type="audio/mpeg",
        headers={"Content-Disposition": "attachment; filename=tts_output.mp3"},
    )

@app.post("/debug/stt")
async def debug_stt(audio: UploadFile = File(...)):
    _require_pipeline()
    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Uploaded audio file is empty.")
    stt_result = await pipeline.stt.transcribe(audio_bytes)
    return stt_result


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.api.server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )
