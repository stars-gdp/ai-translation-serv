# core/pipeline.py

import numpy as np
from transcribers.base import BaseTranscriber
from translators.base import BaseTranslator
from loguru import logger

class AudioProcessingPipeline:
    def __init__(self, translator: BaseTranslator, transcriber: BaseTranscriber):
        self.transcriber = transcriber
        self.translator = translator

    def process(self, chunk: bytes):
        # Конвертируем в numpy (int16 → float32 → torch)
        logger.debug(f"Processing {len(chunk)} bytes of audio")
        audio_np = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0

        transcribed = self.transcriber.transcribe(audio_np)

        return self.translator.translate(transcribed, "rus_Cyrl", "eng_Latn")