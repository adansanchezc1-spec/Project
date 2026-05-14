from abc import ABC, abstractmethod


class NotificationService(ABC):
    @abstractmethod
    def notify(self, subject: str, message: str) -> None:
        raise NotImplementedError
