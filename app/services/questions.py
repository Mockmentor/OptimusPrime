import random

from sqlmodel import Session, select

from app.db import engine
from app.models import Question


def create_question(topic_id: int, text: str) -> Question:
    with Session(engine) as session:
        question = Question(text=text, topic_id=topic_id)

        session.add(question)
        session.commit()
        session.refresh(question)

        return question


def update_question_by_id(
    id: int,
    topic_id: int | None,
    text: str | None,
    audio_file_name: str | None
) -> Question:
    with Session(engine) as session:
        statement = select(Question).where(id=id)
        question = session.exec(statement). one()

        question.text = text
        question.topic_id = topic_id
        question.audio_file_name = audio_file_name

        session.add(question)
        session.commit()
        session.refresh(question)

        return question


def get_questions() -> list[Question]:
    with Session(engine) as session:
        statement = select(Question)
        return session.exec(statement).all()


def get_question_by_id(id: int) -> Question | None:
    with Session(engine) as session:
        return session.get(Question, id)


def get_questions_by_topic_id(topic_id: int) -> list[Question] | None:
    with Session(engine) as session:
        statement = select(Question).where(Question.topic_id == topic_id)
        return session.exec(statement).all()


def get_random_question_by_topic_id(topic_id: int) -> Question | None:
    # questions = get_random_question_by_topic_id(topic_id=topic_id)
    questions = get_questions_by_topic_id(topic_id)
    if questions:
        return random.choice(questions)
