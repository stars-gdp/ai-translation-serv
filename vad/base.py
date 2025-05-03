# vad/base.py

from abc import ABC, abstractmethod
import torch

class BaseVAD(ABC):
    @abstractmethod
    def get_speech_segments(self, wav_tensor: torch.Tensor, sample_rate: int):
        pass
