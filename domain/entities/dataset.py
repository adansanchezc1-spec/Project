from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import pandas as pd

from .cleaning_report import CleaningReport
from .feature import Feature
from domain.enums.dataset_status import DatasetStatus


@dataclass
class Dataset:
    id: str
    name: str
    data_path: str = ""
    status: DatasetStatus = DatasetStatus.NEW
    data: Optional[pd.DataFrame] = None
    features: list[Feature] = field(default_factory=list)
    report: Optional[CleaningReport] = None
    upload_date: datetime = field(default_factory=datetime.now)
    rows: int = 0
    columns: list[str] = field(default_factory=list)
    is_clean: bool = False

    @property
    def path(self) -> str:
        return self.data_path

    def get_dataframe(self) -> Optional[pd.DataFrame]:
        return self.data

    def update_status(self, status: DatasetStatus) -> None:
        self.status = status

    def refresh_profile(self) -> None:
        if self.data is None:
            self.rows = 0
            self.columns = []
            return

        self.rows = len(self.data)
        self.columns = list(self.data.columns)
