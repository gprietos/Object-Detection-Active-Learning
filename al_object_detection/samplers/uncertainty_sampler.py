import numpy as np
import fiftyone as fo
from tqdm import tqdm
from typing import Optional, Union

from .uncertainty_sampler_utils import (
    compute_img_weighted_entropy_uncertainty,
    compute_img_entropy_uncertainty,
    compute_classwise_difficulty,
)


class UncertaintySampler:
    """
    A class to sample a the most uncertain subset of images samples based on object detector predictions.

    Args:
        prediction_field (str, optional): Field name where object detection prediction results (e.g., YOLO) are
            stored in the sample. Defaults to "predictions".

        budget_expansion_ratio (int, optional): hyperparameter that increases the initial query size to ensure
            the intended number of images are queried after the diversity sampling step. Default is set to 5.
            Bigger values will harm the performance in early active learning rounds, as the candidate pools
            might include many samples that the model is certain on. Smaller values will harm the performance
            in later rounds because of the lacking of sample diversity in such small candidate pools.

        method (str, optional): method used to compute the uncertainty of each image. Available methods are
            "entropy" and "weighted_entropy".
            - If set to "entropy", the uncertainty of each image sample is computed as the entropy of the predictions.
                uncertainty = Σ (- p_i * log(p_i) - (1 - p_i) * log(1 - p_i)) / N
              where the sum over all detected objects N and p_i represents the predicted probability of the sample i.
            - If set to "weighted_entropy", the uncertainty of each image sample is computed by weighting the entropy 
              of the predictions with the difficulty of its corresponding class, as defined in class_wise_difficulty_dict
                weighted_uncertainty =  Σ w(cls_i) * (- p_i * log(p_i) - (1 - p_i) * log(1 - p_i))) / N
              where the sum over all detected objects N, p_i represents the predicted probability of the sample
              belonging to class cls_i, and w corresponds to category-wise difficulty coefficient, which is computed
              from the class-wise difficulty values d from class_wise_difficulty_dict.

        class_wise_difficulty_dict (dict, optional): A dictionary of detection difficulties for each class. For
            example d could be d[cls_i] = 1 - val_ap. Detection difficulty is used to raise the importance of
            categories where the prediction model underperformed. Values of the dict must be between 0 and 1.

        alpha (float, optional): Hyperparameter that controls how fast the category-wise difficulty coefficient w
            changes w.r.t. the class-wise difficulty d.

        beta (float, optional): Hyperparameter that controls the upper bound of the category-wise difficulty
            coefficient w.
    """

    def __init__(
        self,
        predictions_field: str = "predictions",
        budget_expansion_ratio: int = 5,
        method: str = "weighted_entropy",
        class_wise_difficulty_dict: Optional[dict] = None,
        alpha: float = 0.3,
        beta: float = 0.2,
    ):
        self.predictions_field = predictions_field
        self.budget_expansion_ratio = budget_expansion_ratio
        self.method = method
        self.d = class_wise_difficulty_dict
        self.alpha = alpha
        self.beta = beta

        self._check_init_args()

    def _check_nonnegative(self, value, desc, strict=True):
        """Validates if value is a valid float > 0 or >=0"""
        if strict:
            negative = (value is None) or (value <= 0)
        else:
            negative = (value is None) or (value < 0)
        if negative or not isinstance(value, (float, np.floating, int, np.integer)):
            raise ValueError("%s should be a nonnegative value. " "%s was given" % (desc, value))

    def _check_init_args(self):
        """Validates the input arguments."""
        if (
            not isinstance(self.budget_expansion_ratio, (float, np.floating, int, np.integer))
            or self.budget_expansion_ratio <= 1
        ):
            raise ValueError(
                "budget_expansion_ratio value should be  greater than 1. %s was given" % (self.budget_expansion_ratio)
            )

        if not isinstance(self.predictions_field, str):
            raise TypeError("predictions_field argument must be a string")

        valid_methods = ["weighted_entropy", "entropy"]
        if self.method not in valid_methods:
            raise ValueError(f"Invalid method '{self.method}'. Must be one of {valid_methods}.")

        # if self.method == "weighted_entropy" and self.d == None:
        #     raise ValueError("If method 'weighted_entropy' is selected, class_wise_difficult_dict must be provided.")

        self._check_nonnegative(self.alpha, "alpha")
        self._check_nonnegative(self.beta, "beta")

        # if self.d is not None and not isinstance(self.d, dict):
        #     raise TypeError("class_wise_difficulty_dict must be a dict with label:difficulty key value pairs")

    def set_prediction_field(self, predictions_field):
        self.predictions_field = predictions_field

    def query(
        self,
        dataset: Union[fo.Dataset, fo.DatasetView],
        budget: int = 100,
        predictions_field: Optional[str] = None,
        verbose: bool = True,
        **kwargs
    ):
        """
        Query a the most uncertain subset of samples from a FiftyOne dataset.

        This method first computes the uncertainty for each sample on the Fiftyone dataset given its predictions
        results (and the class-wise detection difficulty if method 'weighted_entropy' is selected).

        Args:
            dataset(fiftyone.core.view.DatasetView or fiftyone.core.dataset.Dataset):
                The FiftyOne dataset or dataset view pool containing object detection predictions.
            budget (int, optional): The number of uncertain samples to select. Defaults to 100.
            predictions_field (str, optional): Field name where object detection results (e.g., YOLO) are stored in
                the sample.
        Returns:
            fiftyone.core.view.DatasetView: A dataset view containing the most uncertain subset of samples.
        """

        if predictions_field is not None:
            self.predictions_field = predictions_field

        if self.method == "weighted_entropy":
            self.d = compute_classwise_difficulty(
                dataset, predictions_field=self.predictions_field, eval_key=self.predictions_field
            )

        iterable = tqdm(dataset, desc="❓️ Computing samples uncertainty") if verbose else dataset
        for sample in iterable:
            detections = getattr(sample, f"{self.predictions_field}.detections")
            if detections:
                if self.method == "weighted_entropy":
                    conf, cls = zip(*[(detection.confidence, detection.label) for detection in detections])
                    uncertainty = compute_img_weighted_entropy_uncertainty(conf, cls, self.d, self.alpha, self.beta)
                elif self.method == "entropy":
                    conf = [detection.confidence for detection in detections]
                    uncertainty = compute_img_entropy_uncertainty(np.array(conf))
            else:
                uncertainty = 1 + self.beta

            sample["uncertainty"] = uncertainty
            sample.save()

        sorted_uncertainty_samples = dataset.sort_by("uncertainty", reverse=True)

        return sorted_uncertainty_samples.take(budget * self.budget_expansion_ratio)

    @classmethod
    def from_cache_dict(cls, data: dict):
        """Restore an UncertaintySampler instance from cached data."""
        return cls(**data)

    def to_cache_dict(self) -> dict:
        """Return a dictionary representation of the sampler's state."""
        return {
            "predictions_field": self.predictions_field,
            "budget_expansion_ratio": self.budget_expansion_ratio,
            "method": self.method,
            "class_wise_difficulty_dict": self.d,
            "alpha": self.alpha,
            "beta": self.beta,
        }
