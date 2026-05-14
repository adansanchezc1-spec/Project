from presentation.controllers.dataset_controller import DatasetController
from application.services.pipeline_facade import PipelineFacade


class AppView:
    def __init__(self) -> None:
        self.controller = DatasetController(PipelineFacade())

    def render(self, source_path: str) -> None:
        result = self.controller.ingest_and_process(source_path)

        print('=== Dataset Processing Result ===')
        print(f"Dataset ID: {result['dataset'].id}")
        print(f"Name: {result['dataset'].name}")
        print(f"Status: {result['dataset'].status.name}")
        print(f"Features extracted: {len(result['features'])}")
        print(f"Cleaning report: {result['dataset'].report}")
