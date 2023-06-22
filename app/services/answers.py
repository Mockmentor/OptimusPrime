from sqlmodel import Session, select

from app.db import engine
from app.models import Answer


def create_answer(question_id: int) -> Answer:
    with Session(engine) as session:
        answer = Answer(question_id=question_id)
        session.add(answer)
        session.commit()
        session.refresh(answer)
        return answer


def get_answer_by_id(id: int) -> Answer | None:
    with Session(engine) as session:
        return session.get(Answer, id)


def get_answers_by_question_id(question_id: int) -> list[Answer]:
    with Session(engine) as session:
        statement = select(Answer).where(question_id=question_id)
        return session.exec(statement).all()
