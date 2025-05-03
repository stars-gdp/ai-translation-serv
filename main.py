# main.py

from server.socket_server import run_server
from transcribers.whisper_local import WhisperTranscriber
from translators.nllb import NLLBTranslator
from core.pipeline import AudioProcessingPipeline
from loguru import logger
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    # Инициализация компонентов
    logger.info("Initializing modules...")
    transcriber = WhisperTranscriber()
    translator = NLLBTranslator()

    pipeline = AudioProcessingPipeline(translator=translator, transcriber=transcriber)

    # Запуск socket.io сервера
    logger.info("Starting Socket.IO server...")
    run_server(pipeline)

if __name__ == "__main__":
    main()
