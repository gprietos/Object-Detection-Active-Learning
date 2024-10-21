import fiftyone as fo
from pathlib import Path
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import torch
import toml

from typing import Union, Optional
from al_object_detection.object_detectors.base_al_object_detector import BaseALObjectDetector
from al_object_detection.logger import get_logger


class ZeroShot_ALObjectDetector(BaseALObjectDetector):
    def __init__(
        self,
        zero_shot_config: Optional[Union[str, Path, dict]] = None,
    ):
        self.logger = get_logger()

        if isinstance(zero_shot_config, (str, Path)):
            self.config = self._load_config(zero_shot_config)
        elif isinstance(zero_shot_config, dict):
            self.config = zero_shot_config
        else:
            self.logger.error(f"❌🔍 Zero Shot Model configuration not provided / not found at {zero_shot_config}.")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        model_id = self.config["model_id"]
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(self.device)
        self.default_classes = self.config["predict"]["default_classes"]
        self.threshold = self.config["predict"]["threshold"]


    def predict(self, dataset, classes: Optional[list] = None, label_field: Optional[str] = None):

        classes = classes or self.default_classes

        if label_field is None:
            label_field = "al_predictions_round_0"
        if dataset.has_field(label_field):
            self.logger.info(f"📂 Using pre-computed prediction stored in {label_field} field")
        else:
            with fo.ProgressBar() as pb:
                for sample in pb(dataset):

                    image = Image.open(sample.filepath).convert("RGB")

                    inputs = self.processor(images=image, text=classes, return_tensors="pt").to(self.device)

                    with torch.no_grad():
                        outputs = self.model(**inputs)

                    results = self.processor.post_process_object_detection(
                        outputs,
                        threshold=self.threshold,
                        target_sizes=[image.size[::-1]]
                    )[0]

                    detections = []
                    w, h = image.size
                    for label, box, score in zip(results["labels"], results["boxes"], results["scores"]):
                        x1, y1, x2, y2 = box
                        rel_box = [x1 / w, y1 / h, (x2 - x1) / w, (y2 - y1) / h]

                        detections.append(
                            fo.Detection(
                                label=classes[label],
                                bounding_box=rel_box,
                                confidence=score.item()
                            )
                        )

                    sample[label_field] = fo.Detections(detections=detections)
                    sample.save()

    
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
        """Instantiate the Zero Shot Object Detector from a dictionary."""
        return cls(**data)

    def to_cache_dict(self) -> dict:
        """Save the current state of the zero shot object detector in the active learning in a dict."""
        return {"zero_shot_config": self.config}
