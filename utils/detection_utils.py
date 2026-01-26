"""Defines methods for getting the detections from an array of predictions by a model"""
import numpy as np
from typing import List, Tuple

from monipy.data.Detection import Detection


def get_detections(
        ft,
        rec,
    record_id: int,
    predictions: np.ndarray,
    threshold: float,
    event_filter_id: int,
    quality: int,
    con=None,
) -> List[Detection]:
    """
    Creates a list of detection objects from predictions

    Args:
        record_id (int): Record ID as in database.
        predictions (np.ndarray): Array of predictions (Output from model containing probabilities).
        threshold (float): Probability threshold determining when something is labeled 'seizure'.
        event_filter_id (int): Event Filter ID as in database.

    Returns:
        List[Detection]: List of detection objects.

    """

    if predictions is None:
        return []

    detections = apply_threshold(predictions, threshold)

    # print(detections)

    # Chain detections and fill the array to match one entry per 5 min window
    detections = chain_detections(ft,
        record_id, detections, event_filter_id, n_min_detections=3, max_gap=2,quality = quality, con=con
    )

    # Find blocks of positive detections in detection_array
    pos_detection_blocks = _find_positive_detection_blocks(
        detections, n_min_detections=3
    )

    # Return List of instantiated Detection objects
    return _detection_blocks_to_objects(rec,pos_detection_blocks, record_id, con=con)


def apply_threshold(predictions: np.ndarray, threshold: float) -> np.ndarray:
    """
    Convert prediction probabilities to Zero/One (Convert prediction to detection)

    Args:
        predictions (np.ndarray): Array of predictions (Output from model containing probabilities).
        threshold (float): Probability threshold determining when something is labeled 'seizure'.

    Returns:
        predictions (ndarray): Array of 0s and 1s indicating 'no seizure' or 'seizure'.

    """
    return (predictions >= threshold).astype(int)


def chain_detections(
    ft,
    record_id: int,
    detections: np.ndarray,
    event_filter_id: int,
    n_min_detections: int,
    max_gap: int,
    quality: int,
    con=None,
) -> List[int]:
    """
    Chains detections in 0s and 1s array. Short intervals of 1s are removed, and short intervals of 0s
    between 1s are filled, only considering high-quality data.

    Args:
        record_id (int): Record ID as in database.
        detections (np.ndarray): Binary array of model predictions after thresholding.
        event_filter_id (int): Event Filter ID from database.
        n_min_detections (int): Minimum length of valid detection sequence.
        max_gap (int): Maximum allowed 0-gap between 1s to be filled.
        quality (int): Minimum quality threshold.

    Returns:
        np.ndarray: Chained detections array with values in {-1, 0, 1}.
    """
    filter_table = ft.filter_table
    event_filter_column_name = f"event_filter_{event_filter_id}"

    # Apply quality filter first
    quality_mask = filter_table["quality"] >= quality
    # Then apply event filter on quality-filtered rows
    valid_event_mask = quality_mask & filter_table[event_filter_column_name]

    # Initialize full array with -1 (invalid/unusable windows)
    chained_detections = np.full(len(filter_table), -1)

    # Insert model predictions only at valid indices
    valid_idx = filter_table.index[valid_event_mask]
    chained_detections[valid_idx] = detections

    # Fill short gaps of 0s surrounded by 1s (e.g., 1 1 0 1 → 1 1 1 1)
    gap_indices = _find_patterns(chained_detections, max_gap)
    chained_detections = _fill_patterns(chained_detections, gap_indices)

    # Remove short sequences of 1s (likely noise)
    chained_detections = _smooth_short_detections(chained_detections, n_min_detections)

    return chained_detections



def _find_patterns(array: np.ndarray, max_gap: int) -> List[Tuple[int]]:
    """
    Finds fillable 0-gaps between 1s, skipping over any -1 (invalid) entries.

    Returns:
        List[Tuple[int]]: Indices in detections array that need to be filled.
    """
    result = []
    one_indices = np.argwhere(array == 1).flatten()

    for i in range(len(one_indices) - 1):
        start, end = one_indices[i], one_indices[i + 1]
        check_array = array[start:end + 1]

        if len(check_array) == 2:
            continue

        # Skip if any invalid values are in the gap
        if -1 in check_array:
            continue

        if len(check_array) <= 2 + max_gap:
            result.append((start, end))

    return result

def _fill_patterns(array: np.ndarray, gap_indices: List[Tuple[int]]) -> np.ndarray:
    """
    Fill identified gaps in detections with ones

    Args:
        array (np.ndarray): Array of detections (0s and 1s).
        gap_indices (List[Tuple[int]]): Start and end indices of gaps to be filled.

    Returns:
        array (ndarray): Modified detections array.

    """
    for gap_start, gap_end in gap_indices:
        array[gap_start:gap_end] = 1

    return array


def _smooth_short_detections(array: np.ndarray, n_min_detections: int) -> np.ndarray:
    """
    Replace short sequences of 1s with 0s, ignoring regions with -1.
    """
    one_indices = np.argwhere(array == 1).flatten()
    last_corrected_index = -1

    for i in range(len(one_indices)):
        start_idx = one_indices[i]

        if start_idx < last_corrected_index:
            continue

        # Check if we have enough room to evaluate a full window
        if start_idx + n_min_detections > len(array):
            break

        segment = array[start_idx:start_idx + n_min_detections]

        # Skip if the segment contains any invalid values
        if -1 in segment:
            continue

        if np.sum(segment == 1) < n_min_detections:
            array[start_idx:start_idx + n_min_detections] = 0

        last_corrected_index = start_idx + n_min_detections

    return array


def _find_positive_detection_blocks(
    array: np.ndarray, n_min_detections: int
) -> List[int]:
    """
    Identifies detection block start indices from valid detection sequences.

    Ensures the detection block:
    - Contains only `1` values (no `-1`s or `0`s)
    - Has at least `n_min_detections` consecutive valid detections

    Args:
        array (np.ndarray): Array with values in {-1, 0, 1}
        n_min_detections (int): Minimum number of consecutive valid 1s to be considered a block.

    Returns:
        List[int]: Start indices of valid detection blocks.
    """
    i = 0
    block_starts = []

    while i <= len(array) - n_min_detections:
        window = array[i:i + n_min_detections]

        # Ensure the window contains only valid detections
        if np.all(window == 1):
            block_starts.append(i)
            i += 60  # Skip ahead 5 minutes (assuming 5-sec resolution)
        else:
            i += 1

    return block_starts



def _detection_blocks_to_objects(record,
    detection_starts: List[int], record_id: int, con=None
) -> List[Detection]:
    """
    Transform detection start and end indices to Detection objects

    Args:
        detection_starts (List[int]): List of detection start indices.
        record_id (int): Record ID as in database.

    Returns:
        List[Detection]: List of detection objects.

    """

    return [
        Detection(
            record_id,
            record.header["timestamp_start"] + np.timedelta64(5, "s") * detection_start,
            record.header["timestamp_start"]
            + np.timedelta64(5, "s") * detection_start
            + np.timedelta64(300, "s"),
        )
        for detection_start in detection_starts
    ]
