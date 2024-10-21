import fiftyone as fo
from typing import Union


class RandomSampler:
    """
    A class to sample a the most uncertain subset of images samples based on object detector predictions.

    Args: 
        budget_expansion_ratio (int, optional): hyperparameter that increases the initial query size to ensure
            the intended number of images are queried after subsecuent sampling steps. Default is set to 1.
    """

    def __init__(
        self,
        budget_expansion_ratio: int = 1
    ):
        self.budget_expansion_ratio = budget_expansion_ratio


    def query(
        self,
        dataset: Union[fo.Dataset, fo.DatasetView],
        budget: int = 100,
        **kwargs,
    ):
        """
        Query a random subset of samples from a FiftyOne dataset.

        Args:
            dataset(fiftyone.core.view.DatasetView or fiftyone.core.dataset.Dataset):
                The FiftyOne dataset or dataset view pool.
            budget (int, optional): The number of uncertain samples to select. Defaults to 100.
        Returns:
            fiftyone.core.view.DatasetView: A dataset view containing the subset of random samples.
        """

        return dataset.take(budget)

    @classmethod
    def from_cache_dict(cls, data: dict):
        """Restore an RandomSampler instance from cached data."""
        return cls(**data)

    def to_cache_dict(self) -> dict:
        """Return a dictionary representation of the sampler's state."""
        return {"budget_expansion_ratio": self.budget_expansion_ratio}
