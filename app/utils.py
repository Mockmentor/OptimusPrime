import os
from pathlib import Path
from uuid import uuid4

from app.configs import settings


def get_audio_path(audio_file_name: str) -> Path:
    return Path(settings.audio_files_dir, audio_file_name)


def generate_audio_file_name() -> str:
    return '.'.join((str(uuid4()), settings.audio_files_format))


def save_audio(audio: bytes) -> str:
    audio_file_name = generate_audio_file_name()
    path = get_audio_path(audio_file_name)

    with open(str(path), 'wb') as file:
        file.write(audio)

    return audio_file_name


def load_audio(audio_file_name: str) -> bytes:
    path = get_audio_path(audio_file_name)
    return path.read_bytes()


def remove_audio(audio_file_name: str) -> None:
    path = get_audio_path(audio_file_name)

    if not os.path.exists(str(path)):
        return

    os.remove(str(path))
