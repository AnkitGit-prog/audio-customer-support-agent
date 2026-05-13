"""
Audio Pipeline - connects STT -> LLM Agent -> TTS in a single async pipeline.

Usage example:
    pipeline = AudioPipeline(config)
    await pipeline.initialize()

    # Full audio-in / audio-out
    audio_out = await pipeline.process_audio(audio_bytes)

    # Text-in / text-out (no TTS)
    answer = await pipeline.process_text("What is your return policy?")
"""

import logging
from typing import Optional, Dict, Any

from src.stt.base_stt import STTService
from src.llm.agent import CustomerSupportAgent
from src.tts.base_tts import TTSService

logger = logging.getLogger(__name__)


class AudioPipeline:
    """
    End-to-end audio pipeline: Audio -> STT -> LLM+RAG -> TTS -> Audio.

    Orchestrates the three sub-services (STT, LLM agent, TTS) and exposes
    a clean async API for the FastAPI server and tests to consume.
    """

    def __init__(self, stt_config: Dict, llm_config: Dict, tts_config: Dict) -> None:
        """
        Initialise the pipeline with separate configurations for each service.
        """
        self.stt_config = stt_config
        self.llm_config = llm_config
        self.tts_config = tts_config
        self.is_initialized: bool = False

        # Sub-services (created in initialize())
        self.stt: Optional[STTService] = None
        self.llm_agent: Optional[CustomerSupportAgent] = None
        self.tts: Optional[TTSService] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """
        Initialize STT, LLM Agent, and TTS.
        """
        try:
            # Initialize STT
            self.stt = STTService(self.stt_config)
            await self.stt.initialize()

            # Initialize LLM Agent
            self.llm_agent = CustomerSupportAgent(self.llm_config)
            await self.llm_agent.initialize()

            # Initialize TTS
            self.tts = TTSService(self.tts_config)
            await self.tts.initialize()

            self.is_initialized = True
            logger.info("Pipeline initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize pipeline: {e}")
            raise

    async def process_audio(self, audio_bytes: bytes, session_id: str = "default", stream: bool = False, **kwargs) -> Dict[str, Any]:
        """
        Process audio input through the full pipeline.
        Returns a dict with audio_bytes (or generator), transcript, and response_text.
        """
        if not self.is_initialized:
            raise RuntimeError("Pipeline not initialized.")

        # Step 1: STT
        stt_result = await self.stt.transcribe(audio_bytes)
        transcript = stt_result["text"]
        language = stt_result["language"]
        
        # Step 2: LLM
        response_text = await self.llm_agent.process_query(
            query=transcript, 
            session_id=session_id, 
            language=language
        )
        
        # Step 3: TTS
        audio_out = await self.tts.synthesize(
            text=response_text, 
            language=language, 
            stream=stream
        )
        
        return {
            "audio": audio_out,
            "transcript": transcript,
            "response": response_text,
            "language": language
        }

    async def process_text(self, text: str, session_id: str = "default") -> str:
        """
        Process a text query through the LLM agent (no STT/TTS).
        """
        if not self.is_initialized:
            raise RuntimeError(
                "AudioPipeline not initialised. Call initialize() first."
            )

        try:
            logger.info(f"Processing text query for session {session_id}: '{text}'")
            # For text, we default to English unless we add detection
            response: str = await self.llm_agent.process_query(text, session_id=session_id)
            logger.info("Text query processed successfully.")
            return response
        except Exception as exc:
            logger.exception("Pipeline text processing failed: %s", exc)
            return (
                "I encountered a technical error while processing your request. "
                "Please try again or contact our support team."
            )

    async def text_to_audio(self, text: str) -> bytes:
        """
        Convert a text string directly to audio (TTS only, no LLM).

        Args:
            text: The text to synthesise.

        Returns:
            MP3 audio bytes.

        Raises:
            RuntimeError: If the pipeline has not been initialised.
        """
        if not self.is_initialized:
            raise RuntimeError(
                "AudioPipeline not initialised. Call initialize() first."
            )
        return await self.tts.synthesize(text)

    # ------------------------------------------------------------------
    # Status / health
    # ------------------------------------------------------------------

    def get_status(self) -> Dict[str, Any]:
        """
        Return the initialisation status of each sub-service.

        Returns:
            A dict with keys 'stt', 'llm', 'tts', 'pipeline' each mapped
            to a status string ('ready' or 'not_ready').
        """
        return {
            "stt": "ready" if (self.stt and self.stt.is_initialized) else "not_ready",
            "llm": "ready" if (self.llm_agent and self.llm_agent.is_initialized) else "not_ready",
            "tts": "ready" if (self.tts and self.tts.is_initialized) else "not_ready",
            "pipeline": "ready" if self.is_initialized else "not_ready",
        }
