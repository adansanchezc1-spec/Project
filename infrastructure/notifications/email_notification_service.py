from domain.interfaces.notification_service import NotificationService


class EmailNotificationService(NotificationService):
    def notify(self, subject: str, message: str) -> None:
        print(f"[EMAIL] {subject}: {message}")
