from dataclasses import dataclass, field
from typing import List, Optional
import pandas as pd

from .feature import Feature
from .cleaning_report import CleaningReport
from domain.enums.dataset_status import DatasetStatus


@dataclass
class Dataset:
    id: str
    name: str
    data_path: str
    status: DatasetStatus = DatasetStatus.NEW
    data: Optional[pd.DataFrame] = None
    features: List[Feature] = field(default_factory=list)
    report: Optional[CleaningReport] = None
