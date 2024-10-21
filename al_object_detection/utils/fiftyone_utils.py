import os
import yaml
import fiftyone as fo
import fiftyone.utils.random as four

from typing import Optional, Union, List
from pathlib import Path


def add_split_tags(dataset, prop: dict = {"train": 0.8, "val": 0.1, "test": 0.1}):
    no_split_pool = dataset.match(
        ~(
            fo.ViewField("tags").contains("train")
            | fo.ViewField("tags").contains("val")
            | fo.ViewField("tags").contains("test")
        )
    )

    four.random_split(no_split_pool, prop)


def exclude_large_patch_fields(dataset_pool, label_field):
    large_fields = set()
    for sample in dataset_pool:
        detections = getattr(sample, label_field).detections
        for detection in detections:
            for field_name, field_value in detection.iter_fields():
                if len(str(field_value)) > 4096:
                    large_fields.add(field_name)

    for large_field in large_fields:
        dataset_pool = dataset_pool.exclude_fields(f"{label_field}.detections.{large_field}")
    return dataset_pool


def load_yaml(file_path):
    with open(file_path, "r") as stream:
        return yaml.safe_load(stream)


def save_yaml(file_path, data):
    with open(file_path, "w") as file:
        yaml.dump(data, file)


def load_yolo_dataset(
    dataset_name: str,
    dataset_dir: Optional[Union[str, Path]] = None,
    yaml_path: Optional[Union[str, Path]] = None,
    splits: Optional[Union[List[str], str]] = ["train", "val", "test"],
    delete_temp_datasets: Optional[bool] = True,
) -> fo.Dataset:

    if yaml_path is None:
        yaml_path = "dataset.yaml"
    if not os.path.isabs(yaml_path):
        if dataset_dir is None:
            raise ValueError("dataset_dir must be provided if yaml_path is not an absolute path")
        yaml_path = os.path.join(dataset_dir, yaml_path)

    data = load_yaml(yaml_path)
    dataset = fo.Dataset(name=dataset_name)

    def load_and_add_split(ds_name, split, yaml_path, tags):
        split_dataset = fo.Dataset.from_dir(
            name=ds_name,
            yaml_path=yaml_path,
            dataset_type=fo.types.YOLOv5Dataset,
            split=split,
            tags=tags,
        )
        if not dataset.default_classes:
            dataset.default_classes = split_dataset.default_classes
        dataset.add_samples(split_dataset)
        if delete_temp_datasets:
            fo.delete_dataset(ds_name)

    for split in splits:
        if isinstance(data[split], list):
            for sub_split in data[split]:
                sub_split_tag = sub_split.split(os.sep)[0]
                ds_name = f"{dataset_name}_{sub_split_tag}_{split}"
                temp_yaml_path = os.path.join(dataset_dir, f"{ds_name}.yaml")
                temp_yaml = {
                    "path": data["path"],
                    split: sub_split,
                    "nc": data["nc"],
                    "names": data["names"],
                }
                save_yaml(temp_yaml_path, temp_yaml)
                load_and_add_split(ds_name, split, temp_yaml_path, tags=[split, sub_split_tag])
                os.remove(temp_yaml_path)
        else:
            ds_name = f"{dataset_name}_{split}"
            load_and_add_split(ds_name, split, yaml_path, tags=[split])

    return dataset


def export_yolo_dataset(
    dataset: Union[fo.Dataset, fo.DatasetView],
    label_field: str = "ground_truth",
    classes: list = None,
    export_dir: Optional[Union[str, Path]] = None,
    splits: Optional[List[str]] = ["train", "val", "test"],
    export_media: Optional[bool] = True,
) -> None:

    if classes is None or len(classes) == 0:
        if hasattr(dataset, "default_classes") and dataset.default_classes:
            classes = dataset.default_classes
        else:
            raise ValueError("No classes specified and dataset has no default classes.")

    if export_dir is None:
        export_dir = "./exported_dataset"

    for split in splits:
        split_view = dataset.match_tags(split)

        if len(split_view) == 0:
            print(f"[INFO] No samples found in split '{split}'. Skipping export for this split.")
            continue

        try:
            split_view.export(
                dataset_type=fo.types.YOLOv5Dataset,
                classes=classes,
                export_dir=export_dir,
                label_field=label_field,
                export_media=export_media,  # "symlink" to create symlinks to the media files in the output directory
                split=split,
            )

        except Exception as e:
            print(f"[ERROR] Failed to export '{split}' split. Error: {e}")
