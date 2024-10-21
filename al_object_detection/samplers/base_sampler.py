import fiftyone as fo
from typing import Union, Optional
from abc import abstractmethod


class BaseSampler:
    @abstractmethod
    def query(
        self,
        dataset: Union[fo.Dataset, fo.DatasetView],
        budget: int = 100,
        field: Optional[str] = None,
        verbose: bool = True,
    ):
        """
        Abstract method that must be implemented by all sampler subclasses.

        Args:
            dataset (fiftyone.core.view.DatasetView or fiftyone.core.dataset.Dataset): The FiftyOne dataset.
            budget (int): Number of samples to query.
            field (str): An optional field for samplers that need a specific field (e.g., predictions_field)
        Returns:
            fiftyone.core.view.DatasetView: A view of the dataset containing the sampled images.
        """

        pass
