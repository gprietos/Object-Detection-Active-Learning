import fiftyone as fo
from pathlib import Path
import pickle

from typing import Union, Optional

from al_object_detection.active_learner.base_active_learner import BaseActiveLearner
from al_object_detection.samplers import Sampler
from al_object_detection.annotators.annotator import Annotator
from al_object_detection.object_detectors.yolov8_al_object_detector import BaseALObjectDetector
from al_object_detection.logger import get_logger
# from al_object_detection.utils.cache_utils import update_cache


class ObjectDetectionActiveLearner(BaseActiveLearner):
    def __init__(
        self,
        dataset: Union[fo.Dataset, fo.DatasetView],
        budget: int = 100,
        n_rounds: int = 5,
        predictor: Optional[BaseALObjectDetector] = None,
        sampler: Optional[Sampler] = None,
        annotator: Optional[Annotator] = Annotator(), # TODO Change this if more annotators are added
        cache_file: Optional[str] = None,
    ):
        self.dataset = dataset
        self.dataset_name = f"{getattr(dataset, 'name', getattr(dataset, 'dataset_name', ''))}"
        self.budget = budget
        self.n_rounds = n_rounds

        self.predictor = predictor
        self.sampler = sampler
        self.annotator = annotator  

        self.cache_file = Path("cache") / (cache_file or f"{self.dataset_name}_al_state.pkl")

        log_path = Path("logs") / f"logs{self.dataset_name}_al.log".lower()
        self.logger = get_logger(log_file=log_path)

        self.current_round = 0
        self.current_stage = "predicting"
        self.sampled_pool_ids = None


    def save_cache(self):
        """Save the current state of the active learner to the cache file."""
        cache = {
            "dataset_name": self.dataset_name,
            "budget": self.budget,
            "n_rounds": self.n_rounds,
            "model": self.predictor.to_cache_dict(),
            "sampler": self.sampler.to_cache_dict(),
            "annotator": self.annotator.to_cache(),
            "al_round": self.current_round,
            "al_stage": self.current_stage,
            "sampled_pool_ids": self.sampled_pool_ids
        }
        with open(self.cache_file, "wb") as f:
            pickle.dump(cache, f)
        # self.logger.info(f"💾 Cache updated at round {self.current_round}, stage {self.current_stage}")

    def predict(self, dataset):
        """Run predictions on the dataset"""
        self.predictor.predict(dataset, self.predictions_field, self.current_round+1)

    def sample(self, dataset):
        """Sample a pool based on predictions"""
        return self.sampler.query(dataset, self.budget, predictions_field=self.predictions_field)

    def annotate(self, pool):
        """Annotate the sampled pool"""
        self.annotator.annotate(pool, self.predictions_field, self.current_round+1)
        self.logger.info("📤 Dataset uploaded correctly.", extra={"color": "yellow"})
    
    def load_annotations(self, pool):
        self.annotator.wait_for_annotations(self.predictions_field)
        self.annotator.load_annotations(pool)
        self.logger.info("📥 Dataset annotation downloaded and loaded correctly.", extra={"color": "yellow"})

    def train(self, dataset):
        """Train the model with the labelled data"""
        self.predictor.train(dataset, al_round=self.current_round+1)

    def run_active_learning(self):
        try:
            for al_round in range(self.current_round, self.n_rounds):
                self.current_round = al_round
                self.predictions_field = f"al_predictions_round_{al_round + 1 }"

                al_round_text = f"🚀 Object Detection Active Learning Round {al_round + 1 }/{self.n_rounds} 🚀"
                self.logger.info(
                    f"\n{'='*(len(al_round_text)+2)}\n{al_round_text}\n{'='*(len(al_round_text)+2)}",
                    extra={"color": "magenta", "attrs": ["bold"]},
                )

                # Predicting
                if self.current_stage == "predicting":
                    self.logger.info(f"➡️ Round {al_round + 1 }/{self.n_rounds}: 🔮️ Predict", extra={"color": "cyan"})
                    unlabelled_pool = self.dataset.match_tags("unlabelled")
                    self.predict(unlabelled_pool)
                    self.current_stage = "sampling"
                    self.save_cache()

                # Sampling
                if self.current_stage == "sampling":
                    self.logger.info(f"➡️ Round {al_round + 1 }/{self.n_rounds}: 📊 Sampling", extra={"color": "cyan"})
                    unlabelled_pool = self.dataset.match_tags("unlabelled")
                    sampled_pool = self.sample(unlabelled_pool)
                    self.sampled_pool_ids = sampled_pool.values("id")
                    self.current_stage = "annotating-upload"
                    self.save_cache()
                else:
                    if self.sampled_pool_ids:
                        # Restore the sampled pool using the stored IDs
                        sampled_pool = self.dataset[self.sampled_pool_ids]
                    else:
                        raise ValueError("❌ No sampled_pool IDs found.")

                # Annotating
                if self.current_stage == "annotating-upload" or self.current_stage == "annotating-download":
                    self.logger.info(f"➡️ Round {al_round + 1 }/{self.n_rounds}: 📝️ Annotation", extra={"color": "cyan"})
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
                    self.logger.info(f"➡️ Round {al_round + 1 }/{self.n_rounds}: 💪️ Training", extra={"color": "cyan"})
                    self.train(self.dataset)
                    self.current_stage = "predicting"
                    self.current_round += 1
                    self.save_cache()

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
            n_rounds = cache["n_rounds"]
            predictor = cache["model"]["class"].from_cache_dict(cache["model"]["args"])
            sampler = Sampler.from_cache_dict(cache["sampler"])
            annotator = Annotator.from_cache(dataset, cache["annotator"])
            al_round = cache["al_round"]
            al_stage = cache["al_stage"]
            sampled_pool_ids = cache.get('sampled_pool_ids')

            learner = cls(dataset=dataset,budget=budget,n_rounds=n_rounds,predictor=predictor,sampler=sampler,annotator=annotator)
            learner.current_round = al_round
            learner.current_stage = al_stage
            learner.sampled_pool_ids = sampled_pool_ids

            learner.logger.info(
                "✅️ Cache data restored correctly. Resuming Active Learning process.", extra={"color": "yellow"}
            )

            return learner

    def delete_cache(self):
        if Path(self.cache_file).exists():
            Path(self.cache_file).unlink()
            self.logger.info(f"🗑️ Cache file {self.cache_file} deleted.")
