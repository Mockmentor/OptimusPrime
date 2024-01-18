from uuid import UUID

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
# from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware

from app.models import Answer, Message, Question, Room, Topic
from app.services import (
    create_answer,
    create_message,
    create_question,
    create_room,
    create_topic,
    get_answers,
    get_message_by_uuid,
    get_message_similarity,
    get_question_by_id,
    get_questions,
    get_random_question_by_topic_id,
    get_room_by_uuid,
    get_topics,
    update_question_by_id,
    get_questions_by_topic_id,
)
from app.utils import load_audio

from .schemas import (
    CreateAnswerRequest,
    CreateQuestionRequest,
    CreateRoomRequest,
    CreateTopicRequest,
    RoomMessage,
    UpdateQuestionRequest,
)

import time
import json
from app.grpc.unicron import get_unicron_stub
from app.grpc.unicron import unicron_pb2 as unicron_grpc_messages

app = FastAPI()
# app = FastAPI(ssl_keyfile="/etc/letsencrypt/live/mockmentor.ru/privkey.pem", ssl_certfile="/etc/letsencrypt/live/mockmentor.ru/fullchain.pem")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    # allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# app.add_middleware(HTTPSRedirectMiddleware)

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


@app.get('/answers/')
async def fetch_answers(question_id: int | None) -> list[Answer]:
    return get_answers(question_id=question_id)


@app.post('/answers/')
async def insert_answer(answer_request: CreateAnswerRequest) -> Answer:
    return create_answer(**answer_request.dict())


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
    questions = get_questions_by_topic_id(room.topic_id)

    while(len(questions) != 0):
        time.sleep(1)
        
        # question = get_random_question_by_topic_id(room.topic_id)
        question = questions.pop()

        answers = get_answers(question_id=question.id)

        message = create_message(room_uuid=room_uuid, text=question.text)
        room_message = RoomMessage(
            uuid=message.uuid,
            text=message.text,
            type="websocket.send"
            # audio=load_audio(message.audio_file_name)
        )

        msg = {}
        msg['mtype'] = "question"
        msg['text'] = message.text
        json_data = json.dumps(msg)
        message = await websocket.send_text(json_data)
        # await websocket.send(room_message.dict())

        ws_message = await websocket.receive()
        
        if("text" in ws_message):
            try:
                text = ws_message['text']
                room_message = RoomMessage(text=text, type="websocket.send")
                message = create_message(
                    uuid=room_message.uuid,
                    text=room_message.text,
                    audio=room_message.audio,
                    room_uuid=room_uuid,
                )
                similarity = get_message_similarity(text, answers)

                msg = {}
                msg['mtype'] = "correctness"
                msg['text'] = f'Your answer\'s correctness: {round(similarity, 3)}'
                json_data = json.dumps(msg)
                message = await websocket.send_text(json_data)
                
                # message = await websocket.send_text(f'Your answer\'s correctness {round(similarity, 3)}')

            except ValueError as err:
                await websocket.send_json({'error': str(err)})
        
        elif("bytes" in ws_message):
            try:
                b_bytes = ws_message['bytes'] # <class 'bytes'>
                request = unicron_grpc_messages.TextifyRequest(audio=b_bytes)
                text_obj = get_unicron_stub().textify(request)

                text = text_obj.text
                # text = str(text_obj)
                msg = {}
                msg['mtype'] = "textify"
                msg['text'] = text
                json_data = json.dumps(msg)
                message = await websocket.send_text(json_data)
                
                room_message = RoomMessage(text=text, type="websocket.send")
                message = create_message(
                    uuid=room_message.uuid,
                    text=room_message.text,
                    audio=room_message.audio,
                    room_uuid=room_uuid,
                )


                similarity = get_message_similarity(text, answers)
                msg = {}
                msg['mtype'] = "correctness"
                msg['text'] = f'Your answer\'s correctness: {round(similarity, 3)}'
                json_data = json.dumps(msg)
                message = await websocket.send_text(json_data)
                # message = await websocket.send_text(f'Your answer\'s correctness {round(similarity, 3)}')

            except ValueError as err:
                await websocket.send_json({'error': str(err)})

    final_text = "You answered all questions! Well done!"
    message = create_message(room_uuid=room_uuid, text=final_text)
    room_message = RoomMessage(
        uuid=message.uuid,
        text=message.text,
        type="websocket.send"
        # audio=load_audio(message.audio_file_name)
    )

    await websocket.send(room_message.dict())
    websocket.close()

