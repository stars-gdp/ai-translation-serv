# server/socket_server.py

import socketio
from loguru import logger

sio = socketio.Server(async_mode='eventlet', cors_allowed_origins='*')
app = socketio.WSGIApp(sio)

pipeline = None  # глобальная ссылка на pipeline

def run_server(pipeline_instance, host='0.0.0.0', port=3000):
    global pipeline
    pipeline = pipeline_instance

    logger.info(f"Server running on http://{host}:{port}")
    import eventlet.wsgi
    eventlet.wsgi.server(eventlet.listen((host, port)), app)

# Пример события подключения
@sio.event
def connect(sid, environ):
    logger.info(f"Client connected: {sid}")

# Пример события получения аудио чанка
@sio.event
def audio_chunk(sid, data):
    logger.info(f"Received audio chunk from {sid}, len: {len(data)} bytes")
    transcribed = pipeline.process(data)
    logger.debug(f"Transcribed text: {transcribed}")
    sio.emit("transcribed", data=transcribed, to=sid)

@sio.event
def message(sid, data):
    logger.info(f"Received message from {sid}: {data}")