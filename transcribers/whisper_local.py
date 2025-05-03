# transcribers/whisper_local.py

import whisper
import numpy as np
from loguru import logger

from transcribers.base import BaseTranscriber

class WhisperTranscriber(BaseTranscriber):
    def __init__(self, model_name="turbo"):
        self.model = whisper.load_model(model_name)

    def transcribe(self, audio_segment: str | np.ndarray) -> str:
        result = self.model.transcribe(audio=audio_segment, language="ru", verbose=False)
        return result.get("text", "")
