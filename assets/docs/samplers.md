# samplers

The Sampler is responsible for selecting samples from a dataset pool based on specific criteria. In the case of active learning, samplers focus on identifying the most informative samples from the unlabeled pool to maximize learning efficiency of the active learning model when they are annotated.

## Overview

- [UncertaintySampler](../../al_object_detection/samplers/uncertainty_sampler.py): samples the most uncertain subset of image samples based on predictions from the object detection model, in other words,he samples the model struggles with the most. It supports two methods:
  - **Entropy**: the uncertainty of each image sample is computed as the entropy of the predictions.
  - **Weighted Entropy**: the uncertainty of each image sample is computed by weighting the entropy of the predictions with the difficulty of its corresponding class. This is done to ephasize challenging categories where the model may underperform, raising the sampling priority of these classes.

- [DiversitySampler](../../al_object_detection/samplers/diversity_sampler.py): samples the most diverse subset of image samples by computing the pairwise similarities between all samples in the dataset using both global and multi-instance similarity metrics. It then clusters the samples using K-Medoids and selects the medoids as the diverse representatives.

- [RandomSampler](../../al_object_detection/samplers/random_sampler.py): samples a random subset of image samples from the dataset.

<details open>
<summary>Usage</summary>

```python
from al_object_detection.samplers import UncertaintySampler, DiversitySampler, RandomSampler, Sampler

# Sampler Example 1
random_sampler = RandomSampler()
sampler = Sampler([random_sampler])

# Sampler Example 2
uncertainty_sampler = UncertaintySampler()
diversity_sampler = DiversitySampler()
sampler = Sampler([uncertainty_sampler, diversity_sampler])
```

</details>



<details open>
<summary>Define a new Sampler</summary>

You can implement your own samplers! To do that you only have to create a new sampler class that inherits from [BaseSampler](../../al_object_detection/samplers/base_sampler.py) and implement its corresponding methods:

```python
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
    ) -> fo.DatasetView:
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

```

Here’s an example of how to implement a new sampler class:

```python
class RandomSampler:
    """
    A class to take random samples out of the dataset pool.

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
            budget (int, optional): The number of random samples to query. Defaults to 100.
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
```

</details>

## TODO  

- Diversity Sampler optimization
