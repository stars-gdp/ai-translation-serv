# transcribers/base.py

from abc import ABC, abstractmethod
import numpy as np

class BaseTranslator(ABC):
    @abstractmethod
    def translate(self, text: str, src_lang: str, tgt_lang: str) -> str:
        pass

    @abstractmethod
    def get_languages(self) -> list[str]:
        pass