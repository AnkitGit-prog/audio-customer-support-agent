"""
Text-to-Speech (TTS) Service using Microsoft Edge TTS (edge-tts library).

This module provides the TTSService class that:
- Uses the free Microsoft Edge TTS neural voices (no API key required)
- Accepts a text string
- Returns synthesised audio bytes in MP3 format
"""

import os
import logging
import asyncio
from typing import Optional, Any, Dict

logger = logging.getLogger(__name__)


class TTSService:
    """
    Text-to-Speech service powered by Microsoft Edge TTS.

    Uses the `edge-tts` Python library which communicates with Microsoft's
    Edge Read Aloud service to generate high-quality neural speech. No API
    key is required - the service is free to use.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialise the TTS service with optional configuration.
        """
        self.config: Dict[str, Any] = config or {}
        self.is_initialized: bool = False
        
        # Default voice
        self.default_voice: str = self.config.get(
            "tts_voice", os.getenv("TTS_VOICE", "en-US-AriaNeural")
        )
        self.voice = self.default_voice

        # Language to voice mapping
        self.voice_map = {
            "en": "en-US-AriaNeural",
            "es": "es-ES-ElviraNeural",
            "fr": "fr-FR-DeniseNeural",
            "de": "de-DE-KatjaNeural",
            "it": "it-IT-ElsaNeural",
            "hi": "hi-IN-SwaraNeural",
            "ja": "ja-JP-NanamiNeural",
            "zh": "zh-CN-XiaoxiaoNeural"
        }

    async def initialize(self) -> None:
        """
        Initialize the service.
        """
        try:
            self.is_initialized = True
            logger.info(f"TTSService initialised with default voice: {self.default_voice}")
        except Exception as e:
            logger.error(f"Failed to initialize TTSService: {e}")
            raise

    async def synthesize(self, text: str, language: str = "en", stream: bool = False) -> Any:
        """
        Synthesize text to speech.
        If stream=True, yields audio chunks.
        """
        if not self.is_initialized:
            raise RuntimeError("TTSService not initialized.")

        try:
            # pyrefly: ignore [missing-import]
            import edge_tts
            
            # Select voice based on language
            voice = self.voice_map.get(language, self.default_voice)
            
            communicate = edge_tts.Communicate(text, voice)
            
            if stream:
                async def chunk_generator():
                    async for chunk in communicate.stream():
                        if chunk["type"] == "audio":
                            yield chunk["data"]
                return chunk_generator()
            else:
                audio_bytes = b""
                async for chunk in communicate.stream():
                    if chunk["type"] == "audio":
                        audio_bytes += chunk["data"]
                return audio_bytes
        except Exception as e:
            logger.error(f"TTS synthesis failed: {e}")
            return b"" if not stream else None

    async def get_available_voices(self) -> list:
        """
        Return a list of available Edge TTS voices.

        Returns:
            List of voice dictionaries with 'Name', 'Locale', 'Gender' keys.
        """
        try:
            import edge_tts  # type: ignore

            voices = await edge_tts.list_voices()
            return [
                {
                    "name": v["ShortName"],
                    "locale": v["Locale"],
                    "gender": v["Gender"],
                }
                for v in voices
            ]
        except Exception as exc:
            logger.exception("Failed to list voices: %s", exc)
            return []
