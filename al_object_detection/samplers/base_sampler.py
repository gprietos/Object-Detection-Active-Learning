import fiftyone as fo
from typing import Union, Optional
from abc import abstractmethod


class BaseSampler:
    @abstractmethod
    def query(
        self,
        dataset: Union[fo.Dataset, fo.DatasetView],
        budget: int,
        **kwargs
    )-> fo.DatasetView:
        """
        Abstract method that must be implemented by all sampler subclasses.

        Args:
            dataset(fiftyone.core.view.DatasetView or fiftyone.core.dataset.Dataset):
                The FiftyOne dataset or dataset view pool.
            budget (int, optional): Number of samples to query.
        Returns:
            fiftyone.core.view.DatasetView: A view of the dataset containing the sampled images.
        """

        pass

    @classmethod
    def from_cache_dict(cls, data: dict):
        """Restore an UncertaintySampler instance from cached data."""
        return cls(**data)
    
    @abstractmethod
    def to_cache_dict(self) -> dict:
        """Return a dictionary representation of the sampler's state."""
        pass