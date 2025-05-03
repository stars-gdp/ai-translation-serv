# main.py

from server.socket_server import run_server
from vad.silero_vad import SileroVAD
from transcribers.whisper_local import WhisperTranscriber
from core.pipeline import AudioProcessingPipeline
from loguru import logger
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    # Инициализация компонентов
    logger.info("Initializing VAD and Transcriber modules...")
    vad = SileroVAD()
    transcriber = WhisperTranscriber()
    pipeline = AudioProcessingPipeline(vad=vad, transcriber=transcriber)

    # Запуск socket.io сервера
    logger.info("Starting Socket.IO server...")
    run_server(pipeline)

if __name__ == "__main__":
    main()
