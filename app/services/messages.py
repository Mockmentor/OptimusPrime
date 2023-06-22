from uuid import UUID

from funcy import lpluck
from sqlmodel import Session

from app.db import engine
from app.grpc.unicron import get_unicron_stub
from app.grpc.unicron import unicron_pb2 as unicron_grpc_messages
from app.models import Answer, Message


def create_message(
    room_uuid: UUID,
    uuid: UUID | None = None,
    text: str | None = None,
    audio: bytes | None = None
) -> Message:
    if not audio and not text:
        raise Exception

    stub = get_unicron_stub()

    # get audio from text
    if not audio:
        request = unicron_grpc_messages.AudilizeRequest(text=text)
        audio = stub.audiolize(request).audio

    # get text from audio
    if not text:
        request = unicron_grpc_messages.TextifyRequest(audio=audio)
        text = stub.textify(request).text

    #audio_file_name = save_audio(audio)

    with Session(engine) as session:
        message = Message(
            uuid=uuid,
            room_uuid=room_uuid,
            text=text,
            audio_file_name='audio_file_name'
        )
        session.add(message)
        session.commit()
        session.refresh(message)
        return message


def get_message_by_uuid(uuid: UUID) -> Message | None:
    with Session(engine) as session:
        return session.get(Message, uuid)


def get_message_similarity(message: Message, answers: list[Answer]) -> float:
    answers_text = lpluck('text', answers)
    request = unicron_grpc_messages.SimilarityRequest(
        text=message.text, answers=answers_text
    )
    response = get_unicron_stub().similarity(request)
    return response.similarity
