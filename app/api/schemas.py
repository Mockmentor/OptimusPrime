from uuid import UUID

from pydantic import BaseModel, ValidationError


class CreateRoomRequest(BaseModel):
    topic_id: int


class CreateTopicRequest(BaseModel):
    name: str


class CreateQuestionRequest(BaseModel):
    topic_id: int
    text: str


class CreateAnswerRequest(BaseModel):
    question_id: int
    text: str


class UpdateQuestionRequest(BaseModel):
    topic_id: int
    text: str


class RoomMessage(BaseModel):
    uuid: UUID | None = None
    text: str | None = None
    audio: bytes | None = None
    type: str

    def validate(self):
        super.validate()
        if not self.text and not self.audio:
            raise ValidationError('text or audio should be present')
