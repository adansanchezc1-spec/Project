from pathlib import Path
from typing import Any

import pandas as pd

from domain.entities.dataset import Dataset
from domain.interfaces.data_repository import DataRepository


class PandasRepository(DataRepository):
    def __init__(self, storage_path: str = "data") -> None:
        self._storage_path = storage_path

    def exists(self, path: str) -> bool:
        return Path(path).exists()

    def load(self, path: str) -> pd.DataFrame:
        return pd.read_csv(path)

    def save(self, path: str | Dataset, data: Any = None) -> None:
        if isinstance(path, Dataset):
            dataset = path
            target_path = dataset.data_path
            data = dataset.get_dataframe()
        else:
            target_path = path

        if data is None:
            raise ValueError("No data provided to save")

        Path(target_path).parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(target_path, index=False)
