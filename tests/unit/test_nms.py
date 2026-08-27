import os
import sys

# Align python path to workspace root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from agents.inference.inference_agent import non_max_suppression


def test_nms_no_overlap() -> None:
    # Set of bounding boxes (x_min, y_min, x_max, y_max, score, class_id)
    # in meter units, completely disjoint
    boxes = [
        (0.0, 0.0, 50.0, 50.0, 0.9, 0),
        (100.0, 100.0, 150.0, 150.0, 0.85, 0),
    ]
    # No suppression should occur as they are far apart
    result = non_max_suppression(boxes, iou_threshold=0.4)
    assert len(result) == 2
    assert result[0] == boxes[0]
    assert result[1] == boxes[1]


def test_nms_overlap_suppression() -> None:
    # Two highly overlapping boxes of class 0
    # box 1: 0, 0 to 50, 50
    # box 2: 10,10 to 60,60. intersection 40x40 = 1600,
    # union 2500+2500-1600 = 3400, IoU = 1600/3400 = 0.47
    boxes = [
        (0.0, 0.0, 50.0, 50.0, 0.7, 0),
        (10.0, 10.0, 60.0, 60.0, 0.95, 0),  # Higher score
    ]
    # Suppression should drop the lower score box since IOU (0.47) > threshold (0.4)
    result = non_max_suppression(boxes, iou_threshold=0.4)
    assert len(result) == 1
    assert result[0] == boxes[1]  # The higher score box is kept


def test_nms_different_classes() -> None:
    # Overlapping boxes but belonging to different target classes
    boxes = [
        (0.0, 0.0, 50.0, 50.0, 0.7, 0),
        (10.0, 10.0, 60.0, 60.0, 0.95, 1),  # Class 1, not Class 0
    ]
    # No suppression should occur across different target classes
    result = non_max_suppression(boxes, iou_threshold=0.4)
    assert len(result) == 2
