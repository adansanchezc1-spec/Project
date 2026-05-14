from domain.interfaces.notification_service import NotificationService


class EmailNotificationService(NotificationService):
    def __init__(self, email: str = "user@example.com") -> None:
        self._email = email

    def notify(self, message: str, subject: str | None = None) -> None:
        subject_line = subject or "Pipeline notification"
        print(f"[EMAIL] To {self._email} | {subject_line}: {message}")
