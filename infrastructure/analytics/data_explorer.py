import numpy as np

from domain.entities.dataset import Dataset


class DataExplorer:
    def __init__(self) -> None:
        self._correlation_matrix = np.array([])

    def summarize(self, dataset: Dataset) -> dict:
        if dataset.data is None:
            return {}

        return {
            "rows": len(dataset.data),
            "columns": len(dataset.data.columns),
            "missing": int(dataset.data.isna().sum().sum()),
        }

    def generate_profile(self, dataset: Dataset) -> dict:
        if dataset.data is None:
            return {}

        numeric_data = dataset.data.select_dtypes(include=[np.number])
        return {
            "summary": self.summarize(dataset),
            "numeric_describe": numeric_data.describe().to_dict() if not numeric_data.empty else {},
        }

    def calculate_correlations(self, dataset: Dataset) -> np.ndarray:
        if dataset.data is None:
            self._correlation_matrix = np.array([])
            return self._correlation_matrix

        numeric_data = dataset.data.select_dtypes(include=[np.number])
        if numeric_data.empty:
            self._correlation_matrix = np.array([])
        else:
            self._correlation_matrix = numeric_data.corr().to_numpy()
        return self._correlation_matrix

    def plot_distribution(self, dataset: Dataset, column: str):
        if dataset.data is None or column not in dataset.data.columns:
            raise ValueError(f"Column is not available for plotting: {column}")

        import matplotlib.pyplot as plt

        figure, axis = plt.subplots()
        dataset.data[column].hist(ax=axis)
        axis.set_title(f"Distribution of {column}")
        axis.set_xlabel(column)
        axis.set_ylabel("Frequency")
        return figure
