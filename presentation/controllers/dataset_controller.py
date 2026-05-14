from application.services.pipeline_facade import PipelineFacade


class DatasetController:
    def __init__(self, pipeline: PipelineFacade) -> None:
        self.pipeline = pipeline

    def ingest_and_process(self, source_path: str) -> dict:
        dataset = self.pipeline.load(source_path)
        cleaned_dataset = self.pipeline.clean(dataset)
        features = self.pipeline.engineer_features(cleaned_dataset)
        self.pipeline.notify(dataset)
        return {
            'dataset': cleaned_dataset,
            'features': features,
        }
