from .base_sampler import BaseSampler
from typing import Union, Optional
import fiftyone as fo
import pickle
from pathlib import Path


class Sampler(BaseSampler):
    def __init__(self, sampler_list):
        self.sampler_list = sampler_list

    def query(
        self,
        dataset: Union[fo.Dataset, fo.DatasetView],
        budget: int = 100,
        **kwargs
    ):
        """
        Args:
            dataset (fiftyone.core.view.DatasetView or fiftyone.core.dataset.Dataset):
                The FiftyOne dataset or dataset view pool containing object detections, global embeddings,
                and patch embeddings.
            budget (int, optional): The number of diverse samples to select. Defaults to 100.
        Returns:
            fiftyone.core.view.DatasetView: A view of the dataset containing the sampled images.
        """
        sampled_view = dataset.view()
        for sampler in self.sampler_list:
            sampled_view = sampler.query(sampled_view, budget, **kwargs)
        return sampled_view

    @classmethod
    def from_cache_dict(cls, data: dict):
        """Restore a Sampler instance and its sampler_list from a dictionary."""
        sampler_list = [sampler_data["class"].from_cache_dict(sampler_data["args"]) for sampler_data in data["sampler_list"]]
        return cls(sampler_list)

    def to_cache_dict(self) -> dict:
        """Save the current state of the Sampler to a dictionary."""
        return {
            "sampler_list": [
                {"class": type(sampler), "args": {**sampler.to_cache_dict()}} for sampler in self.sampler_list
            ]
        }
