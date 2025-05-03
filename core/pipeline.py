# core/pipeline.py

import numpy as np
import torch
from vad.base import BaseVAD
from transcribers.base import BaseTranscriber
from loguru import logger

class AudioProcessingPipeline:
    def __init__(self, vad: BaseVAD, transcriber: BaseTranscriber, sample_rate: int = 16000):
        self.vad = vad
        self.transcriber = transcriber
        self.sample_rate = sample_rate

    def process(self, chunk: bytes):
        # Конвертируем в numpy (int16 → float32 → torch)
        logger.debug(f"Processing {len(chunk)} bytes of audio")
        audio_np = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0

        return self.transcriber.transcribe(audio_np)

        # VAD: получаем временные интервалы речи
        # audio_tensor = torch.from_numpy(audio_np)
        # speech_segments = self.vad.get_speech_segments(audio_tensor, self.sample_rate)
        # if not speech_segments:
        #     logger.info("No speech detected")
        #     return None
        #
        # transcripts = []
        #
        # for seg in speech_segments:
        #     start = seg['start']
        #     end = seg['end']
        #     segment = audio_np[start:end]
        #     logger.debug(f"Transcribing segment {start}–{end}")
        #     text = self.transcriber.transcribe(segment)
        #     transcripts.append(text)
        #
        # self.audio_buffer.clear()  # очищаем после обработки
        #
        # full_transcript = " ".join(transcripts)
        # logger.info(f"Transcript: {full_transcript}")
        # return full_transcript
