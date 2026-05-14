from abc import ABC, abstractmethod


class MDMService(ABC):
    @abstractmethod
    def sync_master_data(self, dataset_id: str) -> bool:
        raise NotImplementedError
