from sqlmodel import create_engine

from app.configs import settings
from app.models import SQLModel

engine = create_engine(settings.db_url, echo=settings.db_echo_sql)

SQLModel.metadata.create_all(engine)

