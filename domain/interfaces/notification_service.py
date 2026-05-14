from abc import ABC, abstractmethod


class INotificationService(ABC):
    """Contract for pipeline notifications."""

    @abstractmethod
    def notify(self, message: str, subject: str | None = None) -> None:
        raise NotImplementedError


class NotificationService(INotificationService):
    """Backward-compatible alias for notification implementations."""
