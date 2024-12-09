import fiftyone as fo
from typing import Union, Optional
from abc import abstractmethod
from pathlib import Path


class BaseALObjectDetector:
    @abstractmethod
    def predict(self,
                dataset,
                label_field: Optional[str] = None,
                al_round: int = 0):
        pass

    @abstractmethod
    def train(self,
              dataset: Union[fo.Dataset, fo.DatasetView],
              export_dir: Union[str, Path, None] = None,
              train_folder_path: Union[str, Path] = None,
              al_round: int = 0
              ):
        pass

    @classmethod
    def from_cache_dict(cls, data: dict):
        """Instantiate the Object Detector from a dictionary."""
        return cls(**data)

    def to_cache_dict(self) -> dict:
        """Save the current state of the object detector in the active learning in a dict."""
        return {"class": type(self), "args": {"model_path": self.model_path, "model_config": self.config}}