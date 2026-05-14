from application.services.ingestion_service import IngestionService
from application.services.cleaning_service import CleaningService
from application.services.feature_engineering_service import FeatureEngineeringService
from application.services.mdm_service import MDMService
from infrastructure.notifications.email_notification_service import EmailNotificationService
from domain.entities.dataset import Dataset


class PipelineFacade:
    def __init__(self) -> None:
        self.ingestion = IngestionService()
        self.cleaning = CleaningService()
        self.feature_engineering = FeatureEngineeringService()
        self.mdm = MDMService()
        self.notification = EmailNotificationService()

    def load(self, path: str) -> Dataset:
        return self.ingestion.load_dataset(path)

    def clean(self, dataset: Dataset) -> Dataset:
        cleaned = self.cleaning.clean(dataset)
        dataset.status = dataset.status
        return cleaned

    def engineer_features(self, dataset: Dataset) -> list:
        dataset.features = self.feature_engineering.engineer(dataset)
        return dataset.features

    def notify(self, dataset: Dataset) -> None:
        subject = f"Dataset {dataset.name} procesado"
        message = f"Estado: {dataset.status.name}. Features: {len(dataset.features)}"
        self.notification.notify(message, subject=subject)
