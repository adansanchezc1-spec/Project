from dataclasses import dataclass


@dataclass
class Feature:
    name: str
    dtype: str
    importance: float = 0.0

    def calculate_importance(self) -> float:
        return self.importance
