from uuid import UUID

from sqlmodel import Session

from app.db import engine
from app.models import Room


def create_room(topic_id: int) -> Room:
    with Session(engine) as session:
        room = Room(topic_id=topic_id)
        session.add(room)
        session.commit()
        session.refresh(room)
        return room


def get_room_by_uuid(uuid: UUID) -> Room | None:
    with Session(engine) as session:
        return session.get(Room, uuid)

