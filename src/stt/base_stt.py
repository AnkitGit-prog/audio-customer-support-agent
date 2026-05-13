"""
Speech-to-Text (STT) Service using OpenAI Whisper (local inference).

This module provides the STTService class that:
- Loads the Whisper 'base' model locally (no API key required)
- Accepts raw audio bytes (.wav format)
- Returns the transcribed text string
"""

import os
import logging
import asyncio
import tempfile
from typing import Optional, Any, Dict

logger = logging.getLogger(__name__)


class STTService:
    """
    Speech-to-Text service powered by OpenAI Whisper running locally.

    The Whisper 'base' model is downloaded automatically on first run and
    cached by the whisper library for subsequent uses.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialise the STT service with optional configuration.

        Args:
            config: Dictionary with optional key 'stt_model' (default: 'base').
        """
        self.config: Dict[str, Any] = config or {}
        self.is_initialized: bool = False
        self.client: Any = None  # whisper model instance
        self._model_name: str = self.config.get(
            "stt_model", os.getenv("STT_MODEL", "base")
        )

    async def initialize(self) -> None:
        """
        Example for Whisper (local):
        """
        try:
            # pyrefly: ignore [missing-import]
            import whisper
            model_name = self.config.get("model", "base")
            self.client = whisper.load_model(model_name)
            self.is_initialized = True
            logger.info(f"Whisper model '{model_name}' loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load whisper model: {e}")
            raise

    async def _trim_silence(self, audio_bytes: bytes) -> bytes:
        """
        Trim silence from start and end of audio.
        """
        try:
            # pyrefly: ignore [missing-import]
            from pydub import AudioSegment
            import io
            
            # Load audio
            audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
            
            # Trim silence (very basic)
            # You could also use webrtcvad here for better results
            trimmed = audio.strip_silence(silence_thresh=-40, padding=100)
            
            # Export back to bytes
            out = io.BytesIO()
            trimmed.export(out, format="wav")
            return out.getvalue()
        except Exception as e:
            logger.warning(f"Silence trimming failed: {e}")
            return audio_bytes

    async def transcribe(self, audio_bytes: bytes, **kwargs) -> Dict[str, str]:
        """
        Transcribe audio and detect language.
        """
        if not self.is_initialized:
            raise RuntimeError("STTService not initialized.")

        try:
            # Step 0: VAD / Silence Trimming
            audio_bytes = await self._trim_silence(audio_bytes)

            # pyrefly: ignore [missing-import]
            import whisper
            import tempfile
            import os

            # Use delete=False for Windows compatibility
            temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            try:
                temp_file.write(audio_bytes)
                temp_file.close()  # Close so Whisper can open it

                # Run transcription
                result = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: self.client.transcribe(temp_file.name)
                )
            finally:
                # Clean up
                if os.path.exists(temp_file.name):
                    os.remove(temp_file.name)

            return {
                "text": result["text"].strip(),
                "language": result.get("language", "en")
            }
        except Exception as e:
            logger.error(f"Error during transcription: {e}")
            return {"text": "", "language": "en"}
