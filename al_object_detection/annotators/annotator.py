import fiftyone as fo
import os
from dotenv import load_dotenv
import time
from termcolor import colored
from alive_progress import alive_bar

from typing import Optional, Union

from ..utils.fiftyone_utils import exclude_large_patch_fields


class Annotator:
    def __init__(self, anno_key=None):
        load_dotenv()
        self.anno_key = anno_key
        self.result = None

    def annotate(
        self, dataset: Union[fo.Dataset, fo.DatasetView], label_field: Optional[str] = None, al_round: int = 0,
    ):
        # Need to exclude large fields, such as embeddings field, if present to send the dataset pool to the
        # annotator (max length per field is 4096)
        dataset = exclude_large_patch_fields(dataset, label_field)
        project_name = f"{getattr(dataset, 'name') or getattr(dataset, 'dataset_name', '')}_AL"

        self.anno_key = f"{project_name}_anno_round_{al_round}"
        if self.anno_key in dataset.list_annotation_runs():
            dataset.delete_annotation_run(self.anno_key)
        self.result = dataset.annotate(
            anno_key=self.anno_key,
            backend="cvat",
            url=os.environ["FIFTYONE_CVAT_URL"],
            project_name=project_name,
            task_name=self.anno_key,
            username=os.environ["FIFTYONE_CVAT_USERNAME"],
            password=os.environ["FIFTYONE_CVAT_PASSWORD"],
            label_field=label_field,
        )

    def wait_for_annotations(self,label_field):
        """Wait for annotations to be completed"""
        with alive_bar(
            0,
            spinner="waves",
            force_tty=True,
            elapsed=True,
            title=colored("⏳️ Waiting for annotations to be completed...", "cyan"),
            bar=False,
            stats=None,
            monitor=False,
            enrich_print=False,
        ) as bar:
            while not all(
                self.result.get_status()[label_field][task]["status"] == "completed"
                for task in self.result.task_ids
            ):
                time.sleep(5)
                bar()

    def load_annotations(self, dataset, dest_field="ground_truth", cleanup=True, tag_samples=True):
        pool = dataset.load_annotation_view(anno_key=self.anno_key)
        if tag_samples:
            pool.tag_samples("labelled")
            pool.untag_samples("unlabelled")
            pool.save()

        dataset.load_annotations(anno_key=self.anno_key, dest_field=dest_field, cleanup=cleanup)
        dataset.delete_annotation_run(self.anno_key)
        self.anno_key = None

    @classmethod
    def from_cache(cls, dataset, anno_key):
        """Restore a Annotator instance"""
        annotator = cls(anno_key)
        if anno_key is not None:
            annotator.result = dataset.load_annotation_results(anno_key,
                                                            url=os.environ["FIFTYONE_CVAT_URL"],
                                                            username=os.environ["FIFTYONE_CVAT_USERNAME"],
                                                            password=os.environ["FIFTYONE_CVAT_PASSWORD"])
        return annotator

    def to_cache(self) -> dict:
        """Save the current state of the Annotator"""
        return self.anno_key
