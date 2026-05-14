from enum import Enum


class DatasetStatus(Enum):
    RAW = "raw"
    VALIDATED = "validated"
    STORED = "stored"
    CLEANING = "cleaning"
    PROFILED = "profiled"
    TRANSFORMED = "transformed"
    UNIFIED = "unified"
    READY = "ready"
    ERROR = "error"

    # Compatibility aliases used by the current services.
    NEW = RAW
    FAILED = ERROR
