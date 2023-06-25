from uuid import UUID

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware

from app.models import Message, Question, Room, Topic
from app.services import (
    create_message,
    create_question,
    create_room,
    create_topic,
    get_answers_by_question_id,
    get_message_by_uuid,
    get_message_similarity,
    get_question_by_id,
    get_questions,
    get_random_question_by_topic_id,
    get_room_by_uuid,
    get_topics,
    update_question_by_id,
)
from app.utils import load_audio

from .schemas import (
    CreateQuestionRequest,
    CreateRoomRequest,
    CreateTopicRequest,
    RoomMessage,
    UpdateQuestionRequest,
)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get('/topics')
async def fetch_topics() -> list[Topic]:
    return get_topics()


@app.post('/topics')
async def insert_topics(topic_request: CreateTopicRequest) -> Topic:
    return create_topic(**topic_request.dict())


@app.get('/questions/')
async def fetch_questions() -> list[Question]:
    return get_questions()


@app.post('/questions/')
async def insert_question(question_request: CreateQuestionRequest) -> Question:
    return create_question(**question_request.dict())


@app.put('/questions/{question_id}')
async def edit_question(
    question_id: int, question_request: UpdateQuestionRequest
) -> list[Topic]:
    if not get_question_by_id(question_id):
        raise Exception
    return update_question_by_id(id=question_id, **question_request.dict())


@app.get('/messages/{message_uuid}')
async def fetch_message(message_uuid: UUID) -> Message:
    return get_message_by_uuid(message_uuid)


@app.post('/rooms')
async def insert_room(room_request: CreateRoomRequest) -> Room:
    return create_room(room_request.topic_id)


@app.websocket('/rooms/{room_uuid}')
async def ws_room(room_uuid: UUID, websocket: WebSocket):
    await websocket.accept()

    room = get_room_by_uuid(room_uuid)
    question = get_random_question_by_topic_id(room.topic_id)
    answers = get_answers_by_question_id(question.id)

    message = create_message(room_uuid=room_uuid, text=question.text)
    room_message = RoomMessage(
        uuid=message.uuid,
        text=message.text,
        audio=load_audio(message.audio_file_name)
    )

    await websocket.send(room_message.dict())

    while True:
        ws_message = await websocket.receive()
        try:
            room_message = RoomMessage(ws_message.dict())
            message = create_message(
                uuid=room_message.uuid,
                text=room_message.text,
                audio=room_message.audio,
                room_uuid=room_uuid
            )
            similarity = get_message_similarity(message, answers)
            message = websocket.send_text(f'You answers correctness {similarity}')

        except ValueError as err:
            await websocket.send_json({'error': str(err)})

