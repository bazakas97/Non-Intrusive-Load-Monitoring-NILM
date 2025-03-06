import numpy as np


def advanced_postprocess_predictions(
    predictions,
    min_duration=0,
    min_energy_value=0
):
    """
    A more advanced approach:
      1) Zero out negatives.
      2) Basic threshold-based filtering for extremely small cycles:
         - If block doesn't meet min_duration or min_energy_value, set to 0.

    """
    # 1) Zero out any negative predictions
    predictions[predictions < 0] = 0

    # 2) Basic threshold-based filtering
    above_threshold = predictions > min_energy_value
    change_points = np.diff(np.concatenate(([0], above_threshold))).astype(bool)

    indices = np.arange(len(predictions))
    group_ids = np.cumsum(change_points)
    groups = {}

    for idx, group_id in zip(indices, group_ids):
        if group_id not in groups:
            groups[group_id] = []
        groups[group_id].append(idx)

    indices_to_zero = []
    for group_id, idx_list in groups.items():
        # Check if this block is "above threshold" or "below"
        # The first sample's value can indicate on/off
        if predictions[idx_list[0]] > min_energy_value:
            duration = len(idx_list)
            min_val = predictions[idx_list].min()
            # If it doesn't meet duration + min_val criteria, zero out
            if not (duration >= min_duration and min_val > min_energy_value):
                indices_to_zero.extend(idx_list)
        else:
            # This entire block is below min_energy_value
            indices_to_zero.extend(idx_list)

    predictions[indices_to_zero] = 0



    return predictions
