# transcribers/base.py

from abc import ABC, abstractmethod
import numpy as np

class BaseTranscriber(ABC):
    @abstractmethod
    def transcribe(self, audio_segment: np.ndarray) -> str:
        pass
