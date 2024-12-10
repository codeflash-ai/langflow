from collections.abc import Callable
from datetime import datetime


def encode_callable(obj: Callable):
    try:
        return obj.__name__
    except AttributeError:
        return str(obj)


def encode_datetime(obj: datetime):
    return obj.strftime("%Y-%m-%d %H:%M:%S %Z")


CUSTOM_ENCODERS = {Callable: encode_callable, datetime: encode_datetime}
