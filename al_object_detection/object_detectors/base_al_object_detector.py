from abc import abstractmethod


class BaseALObjectDetector:
    def __init__():
        pass

    @abstractmethod
    def predict():
        pass

    @abstractmethod
    def train():
        pass

    @classmethod
    def from_cache_dict(cls, data: dict):
        """Instantiate the Object Detector from a dictionary."""
        return cls(**data)

    def to_cache_dict(self) -> dict:
        """Save the current state of the object detector in the active learning in a dict."""
        return {"class": type(self), "args": {"model_path": self.model_path, "model_config": self.config}}