from pydantic import BaseSettings


class Settings(BaseSettings):
    db_url: str
    db_echo_sql: bool = False

    audio_files_dir: str
    audio_files_format: str = 'wav'

    unicron_host: str = '0.0.0.0'
    unicron_port: int = 9000

    class Config:
        env_file = '.env'


settings = Settings()
