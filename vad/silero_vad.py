# vad/silero_vad.py

import torch
from vad.base import BaseVAD

class SileroVAD(BaseVAD):
    def __init__(self):
        self.model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False
        )
        self.get_speech_timestamps, _, _, _, _ = utils

    def get_speech_segments(self, wav_tensor, sample_rate):
        # wav_tensor — это 1D torch.Tensor со значениями [-1.0, 1.0]
        return self.get_speech_timestamps(wav_tensor, self.model, sampling_rate=sample_rate)
