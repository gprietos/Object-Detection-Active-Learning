import numpy as np
import fiftyone as fo
from typing import Optional


def compute_classwise_difficulty(
    dataset: fo.Dataset,
    predictions_field: str = "predictions",
    ground_truth_field: str = "ground_truth",
    eval_key: Optional[str] = None,
):
    """Computes the prediction difficulty of each class from the model evaluation

    Args:
        dataset (fiftyone.core.view.DatasetView or fiftyone.core.dataset.Dataset): The FiftyOne dataset or dataset view pool
            containing object detections prediction and the ground truth
        predictions_field (str, optional): Field name where object detection results (e.g., YOLO) are stored in each
            sample. Defaults to "predictions".
        ground_truth_field (str, optional): Field name where object detection ground truths are stored in each
            the sample. Defaults to "yolo_predictions". Defaults to "ground_truth".
        eval_key (Optional[str], optional): Field name which will be populated with the records of the evaluation of the
            sample's prediction with respect to its ground truth value. Defaults to None.

    Returns:
        dict: A dictionary of detection difficulties for each class. In this case d[cls_i] = 1 - val_ap.
    """

    # Check if the evaluation has been done before
    if eval_key and dataset.has_evaluation(eval_key):
        print(f"📂 Using cached evaluation: {eval_key}")
        results = dataset.load_evaluation_results(eval_key)
    else:
        print("📝 Computing evaluation...")
        results = dataset.evaluate_detections(
            predictions_field,
            ground_truth_field,
            eval_key=eval_key,
        )

    classes = results.classes.tolist()
    precision_metrics = results.metrics(average=None)["precision"]
    val_ap = dict(zip(classes, precision_metrics))

    return {cls: 1 - ap for cls, ap in val_ap.items()}


def compute_img_weighted_entropy_uncertainty(conf, cls, d, alpha=0.3, beta=0.2):
    """Calculates the difficulty calibrated uncertainty of an image given its predictions results and detection difficulty

    Args:
        conf (array-like): Array of confidence scores for each prediction.

        cls (array-like): Array of class indices for each prediction.

        d (dict): A dictionary of detection difficulties for each class. For example d could be
            d[cls_i] = 1 - val_ap. Values of the dict must be between 0 and 1.

        alpha (float): Hyperparameter that controls how fast the category-wise difficulty coefficient w
            changes w.r.t. the class-wise difficulty d.

        beta (float): Hyperparameter that controls the upper bound of the category-wise difficulty
            coefficient w.

    Returns:
        float: Uncertainty of the unlabelled image by summing the entropy of each detected object weighted
            by the corresponding category-wise difficulty coefficient w
    """

    gamma = np.exp(1.0 / alpha) - 1
    compute_weight = lambda v: 1 + alpha * beta * np.log(1 + gamma * v)
    w = {k: compute_weight(v) for k, v in d.items()}  # category-wise difficulty weight coefficient

    return np.sum(
        np.array(
            [
                w[obj_cls] * (-obj_conf * np.log(obj_conf) - (1 - obj_conf) * np.log((1 - obj_conf)))
                for obj_conf, obj_cls in zip(conf, cls)
            ]
        )
        / len(conf)
    )


def compute_img_entropy_uncertainty(conf):
    """Calculates the uncertainty of an image given its predictions results
    Args:
        conf (array-like): Array of confidence scores for each prediction.
    Returns:
        float: Uncertainty of the unlabelled image by summing the entropy of each detected object
    """

    avg_entropy = np.sum(-conf * np.log(conf) - (1 - conf) * np.log((1 - conf))) / len(conf)
    return avg_entropy
