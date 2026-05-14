from domain.entities.dataset import Dataset


class DataExplorer:
    def summarize(self, dataset: Dataset) -> dict:
        if dataset.data is None:
            return {}

        return {
            'rows': len(dataset.data),
            'columns': len(dataset.data.columns),
            'missing': int(dataset.data.isna().sum().sum()),
        }
