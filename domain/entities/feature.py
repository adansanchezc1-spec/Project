from dataclasses import dataclass


@dataclass
class Feature:
    name: str
    dtype: str
    importance: float = 0.0
