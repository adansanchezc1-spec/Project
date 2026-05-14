from domain.entities.dataset import Dataset
from domain.interfaces.mdm_service import MDMService


class PandasMDMService(MDMService):
    def sync_master_data(self, dataset_id: str) -> bool:
        return True
