from enum import Enum


class DatasetStatus(Enum):
    NEW = 'new'
    CLEANING = 'cleaning'
    READY = 'ready'
    FAILED = 'failed'
