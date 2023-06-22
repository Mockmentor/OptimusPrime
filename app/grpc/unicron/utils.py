import grpc

from app.configs import settings

from .unicron_pb2_grpc import UnicronStub


def get_unicron_stub() -> UnicronStub:
    url = f'{settings.unicron_host}:{settings.unicron_port}'
    channel = grpc.insecure_channel(url)
    return UnicronStub(channel)


