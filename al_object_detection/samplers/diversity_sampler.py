import fiftyone as fo
import fiftyone.zoo as foz
import numpy as np
import torch
import itertools
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment
from sklearn_extra.cluster import KMedoids
from typing import Optional, Union

from al_object_detection.logger import get_logger


class DiversitySampler:
    """
    A class to sample a diverse subset of images based on both global and multi-instance similarity metrics.

    This class computes similarity between images, considering both global image embeddings and object-level embeddings
    (multi-instance) and then selects a diverse set of representative samples using K-Medoids clustering.

    Args:
        weight (float, optional): Weight for the multi-instance similarity. The global similarity weight is set
            to (1 - weight). Range 0 to 1. Defaults to 0.7.

        embeddings_field (str, optional): Field name where global embeddings are stored in the sample.
            Defaults to "embeddings". If not present, embeddings will be computed.

        patches_field (str, optional): Field name where object detection results (e.g., YOLO) are stored in
            the sample. Defaults to "predictions".

        patch_embeddings_field (str, optional): Field name where patch-level embeddings (e.g., CLIP embeddings
            for objects) are stored in the sample. Defaults to "patch_embeddings". If not present, patch
            embeddings will be computed.

        embeddings_model (str, optional): Name of the FiftyOne Model Zoo model used to compute embeddings. Defaults to
            "clip-vit-base32-torch"

        device (str, optional): Device for computation, either "cpu" or "cuda". If not specified, the device is
            chosen based on availability. Defaults to None.
    """

    def __init__(
        self,
        weight: float = 0.7,
        embeddings_field: str = "embeddings",
        patches_field: str = "predictions",
        patch_embeddings_field: str = "patch_embeddings",
        embeddings_model: str = "clip-vit-base32-torch",
        device: Optional[str] = None,
    ):
        self.weight = weight
        self.embeddings_field = embeddings_field
        self.patches_field = patches_field
        self.patch_embeddings_field = patch_embeddings_field
        self.embeddings_model = embeddings_model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self._check_init_args()

        self.multi_instance_weight = weight
        self.global_weight = 1 - weight
        self.similarity_metric = torch.nn.CosineSimilarity(dim=0)

        self.logger = get_logger()

    def _check_init_args(self):
        """Validates the input arguments."""
        if not isinstance(self.weight, (float, np.floating, int, np.integer)) and not (
            0 <= self.multi_instance_weight <= 1
        ):
            raise ValueError(f"'weight' should be a nonnegative value between 0 and 1. {self.weight} was given.")

    def _check_embeddings(self, dataset):
        # Check if global embeddings are available
        if self.global_weight > 0:
            no_global_embeddings_view = (
                dataset.match({self.embeddings_field: None}) if dataset.has_field(self.embeddings_field) else dataset
            )
            if no_global_embeddings_view:
                self._compute_global_embeddings(no_global_embeddings_view)

        # Check if patch embeddings are available
        if self.multi_instance_weight > 0:
            patch_embeddings_field = f"{self.patches_field}.detections.{self.patch_embeddings_field}"
            no_patch_embeddings_view = (
                dataset.match({patch_embeddings_field: None}) if dataset.has_field(patch_embeddings_field) else dataset
            )
            if no_patch_embeddings_view:
                self._compute_patch_embeddings(no_patch_embeddings_view)

    def _compute_global_embeddings(self, dataset):
        self.logger.info("🧠 Computing global embeddings")
        dataset.compute_embeddings(
            model=foz.load_zoo_model(self.embeddings_model), embeddings_field=self.embeddings_field, progress=True
        )

    def _compute_patch_embeddings(self, dataset):
        self.logger.info("🧠 Computing patch embeddings")
        dataset.compute_patch_embeddings(
            model=foz.load_zoo_model(self.embeddings_model),
            patches_field=self.patches_field,
            embeddings_field=self.patch_embeddings_field,
            progress=True,
        )

    def _filter_similarity_pairs(self, similarity_matrix, row_ind, col_ind, threshold):
        filtered_pairs = [(r, c) for r, c in zip(row_ind, col_ind) if similarity_matrix[r, c] >= threshold]
        return zip(*filtered_pairs) if filtered_pairs else ([], [])

    def _compute_weighted_similarity_average(self, similarity_matrix, conf1, conf2, row_ind, col_ind):
        total_weighted_similarity = sum(conf1[i] * conf2[j] * similarity_matrix[i, j] for i, j in zip(row_ind, col_ind))
        total_confidence_weight = sum(conf1[i] * conf2[j] for i, j in zip(row_ind, col_ind))
        return total_weighted_similarity / total_confidence_weight if total_confidence_weight > 0 else 0

    def _penalize_similarity_score(self, weighted_similarity, num_matched, num_objects1, num_objects2, penalty=0.4):
        num_unmatched = (num_objects1 + num_objects2) - 2 * num_matched
        unmatched_ratio = num_unmatched / (num_objects1 + num_objects2)
        return weighted_similarity * (1 - penalty * unmatched_ratio)

    def _compute_multi_instance_similarity(self, sample1, sample2, similarity_threshold=0.4, penalty_factor=0.5):
        # Extract class, confidence, and features
        def extract_data(sample):
            return [
                (detection.label, detection.confidence, getattr(detection, self.patch_embeddings_field))
                for detection in getattr(sample, f"{self.patches_field}.detections", [])
            ] or None

        sample1_data = extract_data(sample1)
        sample2_data = extract_data(sample2)

        if not sample1_data or not sample2_data:
            return 0

        cls1, conf1, feats1 = zip(*sample1_data)
        cls2, conf2, feats2 = zip(*sample2_data)

        similarity_matrix = np.zeros((len(cls1), len(cls2)))

        for i, (c1, f1) in enumerate(zip(cls1, feats1)):
            for j, (c2, f2) in enumerate(zip(cls2, feats2)):
                if c1 == c2:
                    similarity_matrix[i, j] = self.similarity_metric(
                        torch.tensor(f1, device=self.device), torch.tensor(f2, device=self.device)
                    ).item()

        row_ind, col_ind = linear_sum_assignment(-similarity_matrix)
        filtered_row_ind, filtered_col_ind = self._filter_similarity_pairs(
            similarity_matrix, row_ind, col_ind, similarity_threshold
        )

        weighted_similarity = self._compute_weighted_similarity_average(
            similarity_matrix, conf1, conf2, filtered_row_ind, filtered_col_ind
        )
        return self._penalize_similarity_score(
            weighted_similarity, len(filtered_row_ind), len(cls1), len(cls2), penalty_factor
        )

    def _compute_similarity_matrix(self, H: Union[fo.Dataset, fo.DatasetView], verbose: bool = False):
        n = len(H)
        similarity_matrix = np.ones((n, n))
        sample_ids = {sample.id: i for i, sample in enumerate(H)}

        combinations = itertools.combinations(H, 2)
        if verbose:
            combinations = tqdm(combinations, total=(n * (n - 1)) // 2, desc="🔢 Computing similarity matrix")

        for sample1, sample2 in combinations:
            # Global Similarity
            global_similarity = (
                self.similarity_metric(
                    torch.tensor(getattr(sample1, self.embeddings_field), device=self.device),
                    torch.tensor(getattr(sample2, self.embeddings_field), device=self.device),
                ).item()
                if self.global_weight > 0
                else 0
            )

            # Multi Instance Similarity
            multi_instance_similarity = (
                self._compute_multi_instance_similarity(sample1, sample2) if self.multi_instance_weight > 0 else 0
            )

            similarity = (
                self.multi_instance_weight * multi_instance_similarity + self.global_weight * global_similarity
            ) / (self.multi_instance_weight + self.global_weight)

            i, j = sample_ids[sample1.id], sample_ids[sample2.id]
            similarity_matrix[i, j] = similarity_matrix[j, i] = similarity

        return similarity_matrix, sample_ids

    def _get_medoid_indices(self, similarity_matrix, n_clusters):
        kmedoids = KMedoids(
            n_clusters=n_clusters, metric="precomputed", method="pam", init="k-medoids++", random_state=42
        )
        kmedoids.fit(1 - similarity_matrix)
        return kmedoids.medoid_indices_

    def set_patches_field(self, patches_field: str):
        self.patches_field = patches_field

    def query(
        self,
        dataset: Union[fo.Dataset, fo.DatasetView],
        budget: int = 100,
        predictions_field: Optional[str] = None,
        verbose: bool = True,
        **kwargs
    ):
        """
        Query a diverse subset of samples from a FiftyOne dataset based on similarity.

        This method computes the pairwise similarities between all samples in the input FiftyOne
        dataset or dataset view using both global and multi-instance similarity metrics. It then
        clusters the samples using K-Medoids and selects the medoids as the diverse representatives.

        Args:
            dataset (fiftyone.core.view.DatasetView or fiftyone.core.dataset.Dataset):
                The FiftyOne dataset or dataset view pool containing object detections, global embeddings,
                and patch embeddings.
            budget (int, optional): The number of diverse samples to select. Defaults to 100.
            predictions_field (str, optional): Field name where object detection results (e.g., YOLO) are stored in
                the sample.

        Returns:
            fiftyone.core.view.DatasetView: A dataset view containing the diverse subset of samples.
        """
        if predictions_field is not None:
            self.patches_field = predictions_field

        self._check_embeddings(dataset)

        similarity_matrix, sample_ids = self._compute_similarity_matrix(dataset, verbose=verbose)
        medoid_indices = self._get_medoid_indices(similarity_matrix, budget)
        medoid_ids = [k for idx in medoid_indices for k, v in sample_ids.items() if v == idx]
        return dataset[medoid_ids]

    @classmethod
    def from_cache_dict(cls, data: dict):
        """Restore a DiversitySampler instance from cached data."""
        return cls(**data)

    def to_cache_dict(self) -> dict:
        """Return only the constructor arguments to be cached."""
        return {
            "weight": self.weight,
            "embeddings_field": self.embeddings_field,
            "patches_field": self.patches_field,
            "patch_embeddings_field": self.patch_embeddings_field,
            "embeddings_model": self.embeddings_model,
            "device": self.device,
        }
