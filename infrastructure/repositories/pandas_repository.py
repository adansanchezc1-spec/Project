import pandas as pd
from domain.interfaces.data_repository import DataRepository


class PandasRepository(DataRepository):
    def load(self, path: str) -> pd.DataFrame:
        return pd.read_csv(path)

    def save(self, path: str, data: pd.DataFrame) -> None:
        data.to_csv(path, index=False)
