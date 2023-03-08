from typing import Any, Dict, List, Optional


def batch2list(data: Dict[str, List]) -> List[Dict[str, Any]]:
    """Convert from a batched target dict to list of normal target dicts

    ex:
    {"image_batch": image_batch, "bboxes_batch": bboxes_batch, ...}
    =>
    [
        {"image": image_batch[0], "bboxes": bboxes_batch[0], ...},
        {"image": image_batch[1], "bboxes": bboxes_batch[1], ...},
        ...
    ]
    """
    if "image_batch" not in data:
        raise ValueError("Batch-based transform should have `image_batch` target")
    batch_size = len(data["image_batch"])
    items = []
    for i in range(batch_size):
        item = {}
        for k, v in data.items():
            if k.endswith("_batch"):
                # ex. {"image_batch": image_batch} -> {"image": image_batch[i]}
                item_k = to_unbatched_name(k)
                item[item_k] = v[i]
            else:
                raise ValueError(f"All key must have '_batch' suffix, got `{k}`")
        items.append(item)
    return items


def list2batch(data: List[Dict[str, Any]]) -> Dict[str, List]:
    """Convert from a list of normal target dicts to a batched target dict

    ex:
    [
        {"image": image_batch[0], "bboxes": bboxes_batch[0], ...},
        {"image": image_batch[1], "bboxes": bboxes_batch[1], ...},
        ...
    ]
    =>
    {"image_batch": image_batch, "bboxes_batch": bboxes_batch, ...}
    """

    if len(data) == 0:
        raise ValueError("The input should have at least one item.")

    item = data[0]
    batch: Dict[str, Any] = {f"{k}_batch": [] for k in item.keys()}
    for item in data:
        for k, v in item.items():
            batch_k = to_batched_name(k)
            batch[batch_k].append(v)

    return batch


def to_unbatched_name(batched_name: str) -> str:
    """Get a normal target name from a batched target name

    If the given name does not have "_batched" suffix, ValueError will be raised.
    ex. `abc --> abc_batched`
    """
    if not batched_name.endswith("_batch"):
        raise ValueError(f"Batched target name must have '_batch' suffix, got `{batched_name}`")
    return batched_name.replace("_batch", "")


def to_batched_name(name: str) -> str:
    """Get a unbatched target name from a normal target name

    If the given name already has had "_batched" suffix, ValueError will be raised.
    ex. `abc_batched --> abc `
    """

    if name.endswith("_batch"):
        raise ValueError(f"Non batched target name must not have '_batch' suffix, got `{name}`")
    return f"{name}_batch"


def concat_batches(batches: List[Dict[str, List]]) -> Dict[str, List]:
    """Concatenate batched targets
     ex:
      [
        {"image_batch": image_batch1, "bboxes_batch": bboxes_batch1, ...}
        {"image_batch": image_batch1, "bboxes_batch": bboxes_batch1, ...}
      ]
    =>
      {
        "image_batch": image_batch1 + image_batch2, "bboxes_batch": bboxes_batch1 + bboxes_batch2, ...
      }
    """

    n_batches = len(batches)
    if n_batches == 0:
        raise ValueError("The input should have at least one item.")

    keys = list(batches[0].keys())
    out_batch: Dict[str, List] = {k: [] for k in keys}
    for batch in batches:
        for k in keys:
            for item in batch[k]:
                out_batch[k].append(item)
    return out_batch
