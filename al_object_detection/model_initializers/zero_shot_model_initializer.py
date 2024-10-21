import fiftyone as fo
from pathlib import Path
import pickle

from typing import Union, Optional

from al_object_detection.samplers import Sampler
from al_object_detection.annotators.annotator import Annotator
from al_object_detection.object_detectors.yolov8_al_object_detector import BaseALObjectDetector
from al_object_detection.model_initializers.zero_shot_al_object_detector import ZeroShot_ALObjectDetector
from al_object_detection.logger import get_logger


class ZeroShotModelInitializer():
    def __init__(
        self,
        dataset: Union[fo.Dataset, fo.DatasetView],
        budget: int = 300,
        zeroshot_model: Optional[ZeroShot_ALObjectDetector] = None, 
        al_model: Optional[BaseALObjectDetector] = None,
        sampler: Optional[Sampler] = None,
        annotator: Optional[Annotator] = Annotator(), # TODO Change this if more annotators are added
        cache_file: Optional[str] = None,
    ):
        self.dataset = dataset
        self.dataset_name = f"{getattr(dataset, 'name', getattr(dataset, 'dataset_name', ''))}"
        self.budget = budget

        self.zeroshot_model = zeroshot_model
        self.predictor = al_model
        self.sampler = sampler
        self.annotator = annotator  

        self.cache_file = cache_file or f"{self.dataset_name}_zero_shot_initializer.pkl"

        self.logger = get_logger(log_file=f"{self.dataset_name}_zero_shot_initializer.log".lower())

        self.current_stage = "sampling"
        self.sampled_pool_ids = None


    def save_cache(self):
        """Save the current state of the active learner to the cache file."""
        cache = {
            "dataset_name": self.dataset_name,
            "budget": self.budget,
            "zeroshot_model": self.zeroshot_model.to_cache_dict(),
            "al_model": self.predictor.to_cache_dict(),
            "sampler": self.sampler.to_cache_dict(),
            "annotator": self.annotator.to_cache(),
            "sampled_pool_ids": self.sampled_pool_ids
        }
        with open(self.cache_file, "wb") as f:
            pickle.dump(cache, f)

    def sample(self, dataset):
        """Sample a pool based on predictions"""
        return self.sampler.query(dataset, self.budget)
    
    def zero_shot_predict(self, dataset):
        """Run predictions on the dataset"""
        self.zeroshot_model.predict(dataset)

    def annotate(self, pool):
        """Annotate the sampled pool"""
        self.annotator.annotate(pool, 0, "al_predictions_round_0")
        self.logger.info("📤 Dataset uploaded correctly.", extra={"color": "yellow"})
    
    def load_annotations(self, pool):
        self.annotator.wait_for_annotations("al_predictions_round_0")
        self.annotator.load_annotations(pool)
        self.logger.info("📥 Dataset annotation downloaded and loaded correctly.", extra={"color": "yellow"})

    def train(self, dataset):
        """Train the model with the labelled data"""
        self.predictor.train(dataset, al_round=0)

    def run_model_initialization(self):
        al_round_text = f"🚀 Object Detector Initialization from Zero-Shot Predictions 🚀"
        self.logger.info(
            f"\n{'='*(len(al_round_text)+2)}\n{al_round_text}\n{'='*(len(al_round_text)+2)}",
            extra={"color": "magenta", "attrs": ["bold"]},
        )

        try:
            # Sampling
            if self.current_stage == "sampling":
                self.logger.info(f"➡️ Round 0: 📊 Sampling", extra={"color": "cyan"})
                unlabelled_pool = self.dataset.match_tags("unlabelled")
                sampled_pool = self.sample(unlabelled_pool)
                self.sampled_pool_ids = sampled_pool.values("id")
                self.current_stage = "predicting"
                self.save_cache()
            else:
                if self.sampled_pool_ids:
                    # Restore the sampled pool using the stored IDs
                    sampled_pool = self.dataset[self.sampled_pool_ids]
                else:
                    raise ValueError("❌ No sampled_pool IDs found.")

            # Predicting
            if self.current_stage == "predicting":
                self.logger.info(f"➡️ Round 0: 🔮️ Zero Shot Prediction", extra={"color": "cyan"})
                self.zero_shot_predict(sampled_pool)
                self.current_stage = "annotating-upload"
                self.save_cache()

            # Annotating
            if self.current_stage == "annotating-upload" or self.current_stage == "annotating-download":
                self.logger.info(f"➡️ Round 0: 📝️ Annotation", extra={"color": "cyan"})
                if self.current_stage == "annotating-upload":
                    self.annotate(sampled_pool)
                    self.current_stage = "annotating-download"
                    self.save_cache()

                if self.current_stage == "annotating-download":
                    self.load_annotations(sampled_pool)
                    self.current_stage = "training"
                    self.save_cache()

            # Training
            if self.current_stage == "training":
                self.logger.info(f"➡️ Round 0: 💪️ Training", extra={"color": "cyan"})
                self.train(self.dataset)

            self.delete_cache()

        except KeyboardInterrupt:
            self.logger.warning(
                "❌ Process interrupted. You can resume it later using the from_cache method", extra={"color": "red"}
            )

    @classmethod
    def from_cache(cls, cache_file: str):
        if Path(cache_file).exists():
            with open(cache_file, "rb") as f:
                cache = pickle.load(f)
            dataset = fo.load_dataset(cache["dataset_name"])
            budget = cache["budget"]
            zero_shot_model = ZeroShot_ALObjectDetector.from_cache_dict(cache["zero_shot_model"])
            al_model = cache["al_model"]["class"].from_cache_dict(cache["al_model"]["args"])
            sampler = Sampler.from_cache_dict(cache["sampler"])
            annotator = Annotator.from_cache(dataset, cache["annotator"])
            sampled_pool_ids = cache.get('sampled_pool_ids')

            learner = cls(dataset=dataset,budget=budget,zero_shot_model=zero_shot_model,al_model=al_model,sampler=sampler,annotator=annotator)
            learner.sampled_pool_ids = sampled_pool_ids

            learner.logger.info(
                "✅️ Cache data restored correctly. Resuming Active Learning process.", extra={"color": "yellow"}
            )

            return learner
        
    def delete_cache(self):
        if Path(self.cache_file).exists():
            Path(self.cache_file).unlink()
            self.logger.info(f"🗑️ Cache file {self.cache_file} deleted.")