from pathlib import Path

from domain.entities.dataset import Dataset
from domain.enums.dataset_status import DatasetStatus
from domain.interfaces.data_repository import IDataRepository
from domain.interfaces.notification_service import INotificationService
from infrastructure.notifications.email_notification_service import EmailNotificationService
from infrastructure.repositories.pandas_repository import PandasRepository


class IngestionService:
    def __init__(
        self,
        repository: IDataRepository | None = None,
        notification_service: INotificationService | None = None,
    ) -> None:
        self._repository = repository or PandasRepository()
        self._notification_service = notification_service or EmailNotificationService()

    def load_dataset(self, path: str) -> Dataset:
        if not self._repository.exists(path):
            self._notification_service.notify(
                f"Dataset path does not exist: {path}",
                subject="Dataset ingestion failed",
            )
            return Dataset(
                id=path,
                name=Path(path).name,
                data_path=path,
                status=DatasetStatus.ERROR,
            )

        dataframe = self._repository.load(path)
        dataset = Dataset(
            id=path,
            name=Path(path).name,
            data_path=path,
            status=DatasetStatus.STORED,
            data=dataframe,
        )
        dataset.refresh_profile()
        return dataset
