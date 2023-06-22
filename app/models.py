from uuid import UUID, uuid4

from sqlmodel import Field, SQLModel


class Topic(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    name: str


class Question(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    text: str
    topic_id: int = Field(foreign_key='topic.id')


class Answer(SQLModel, table=True):
    id: int | None = Field(default=None,primary_key=True)
    text: str
    question_id: int = Field(foreign_key='question.id')


class Message(SQLModel, table=True):
    uuid: UUID = Field(default_factory=uuid4, primary_key=True)
    text: str
    audio_file_name: str
    room_uuid: UUID = Field(foreign_key='room.uuid')


class Room(SQLModel, table=True):
    uuid: UUID = Field(default_factory=uuid4, primary_key=True)
    topic_id: int = Field(foreign_key='topic.id')

