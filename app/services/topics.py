from sqlmodel import Session, select

from app.db import engine
from app.models import Topic


def create_topic(name: str) -> Topic:
    with Session(engine) as session:
        topic = Topic(name=name)
        session.add(topic)
        session.commit()
        session.refresh(topic)
        return topic


def get_topics() -> list[Topic]:
    with Session(engine) as session:
        statement = select(Topic)
        return session.exec(statement).all()


def get_topic_by_id(id: int) -> Topic | None:
    with Session(engine) as session:
        return session.get(Topic, id)
