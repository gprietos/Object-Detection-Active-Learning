import fiftyone as fo
from ultralytics import YOLO
from pathlib import Path
import toml

from typing import Union, Optional
from al_object_detection.object_detectors.base_al_object_detector import BaseALObjectDetector
from al_object_detection.utils.fiftyone_utils import export_yolo_dataset, add_split_tags
from al_object_detection.logger import get_logger


class YOLOv8_ALObjectDetector(BaseALObjectDetector):
    def __init__(
        self,
        model_path: Union[str, Path] = "yolov8l.pt",
        model_config: Optional[Union[str, Path, dict]] = None,
    ):
        self.logger = get_logger()

        self.model_path = Path(model_path)
        if Path.exists(self.model_path):
            self.model = YOLO(self.model_path)
        else:
            self.logger.error(f"❌🔍 Model weights not found at {self.model_path}.")

        if isinstance(model_config, (str, Path)):
            self.config = self._load_config(model_config)
        elif isinstance(model_config, dict):
            self.config = model_config

    def predict(self, dataset, label_field: Optional[str] = None, al_round: int = 0):
        if label_field is None:
            label_field = f"al_predictions_round_{al_round}"
        if dataset.has_field(label_field):
            self.logger.info(f"📂 Using pre-computed prediction stored in {label_field} field")
        else:
            # unlabelled_pool = dataset.match_tags("unlabelled")
            # unlabelled_pool.apply_model(self.model, label_field=label_field, **self.config["predict"])
            dataset.apply_model(self.model, label_field=label_field, **self.config["predict"])

    def train(
        self,
        dataset: Union[fo.Dataset, fo.DatasetView],
        export_dir: Union[str, Path, None] = None,
        train_folder_path: Union[str, Path] = None,
        al_round: int = 0,
    ):

        if train_folder_path is not None and isinstance(train_folder_path, str):
            training_path = Path(train_folder_path)
        else:
            project_name = f"{getattr(dataset, 'name') or getattr(dataset, 'dataset_name', '')}_AL"
            training_path = Path.cwd() / "Trainings" / Path(project_name)

        if export_dir is None:
            export_dir = str(Path.cwd() / "exported_datasets" / project_name)

        # Check if last.pt exists to resume training if previous training was interrupted
        last_checkpoint_path = training_path / f"AL_round_{al_round}" / "weights" / "last.pt"
        resume_training = last_checkpoint_path.exists()

        if resume_training:
            self.model = YOLO(last_checkpoint_path)
            self.logger.info(f"📂 Resuming training from {last_checkpoint_path}")
            self.model.train(resume=True)
        else:
            # TODO Export only newly added samples?
            self.export_dataset(dataset, export_dir=export_dir)

            self.model.train(
                project=training_path,
                name=f"AL_round_{al_round}",
                data=Path(export_dir) / "dataset.yaml",
                **self.config["train"],
            )

        self.model_path = training_path / f"AL_round_{al_round}" / "weights" / "best.pt"

    def export_dataset(
        self,
        dataset,
        label_field="ground_truth",
        classes: Optional[list] = None,
        export_dir: Optional[str] = None,
    ):

        labelled_pool = dataset.match_tags("labelled")
        add_split_tags(labelled_pool)
        export_yolo_dataset(
            dataset=labelled_pool,
            label_field=label_field,
            classes=classes,
            export_dir=export_dir,
        )

    def _load_config(self, config_path=None, **kwargs):
        """Load configuration from a TOML file or kwargs. Kwargs will override TOML file."""
        config = {}
        if config_path:
            with open(config_path, "r") as f:
                config = toml.load(f)
        config.update(kwargs)  # Override config with kwargs if any
        return config

    @classmethod
    def from_cache_dict(cls, data: dict):
        """Instantiate the Object Detector from a dictionary."""
        return cls(**data)

    def to_cache_dict(self) -> dict:
        """Save the current state of the object detector in the active learning in a dict."""
        return {"class": type(self), "args": {"model_path": self.model_path, "model_config": self.config}}
