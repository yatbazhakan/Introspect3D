from utils.boundingbox import BoundingBox
import numpy as np
def create_bounding_boxes_from_predictions(boxes: np.ndarray):
    """Creates a list of BoundingBox objects from a numpy array of boxes."""
    #TODO: checks for the boxes in terms of shape (what if more values are given)
    bounding_boxes = []
    for box in boxes:
        if isinstance(box,BoundingBox):
            print("Box is already a BoundingBox object")
            bounding_boxes.append(box)
            continue
        center = box[:3]
        dimensions = box[3:6]
        rotation = box[6]

        center[2] += dimensions[2]/2
        bounding_box = BoundingBox(center, dimensions, rotation, 0)
    
        bounding_boxes.append(bounding_box)
    return bounding_boxes

def check_detection_matches(ground_truth_boxes, predicted_boxes, iou_threshold:float=0.5):
    """Checks if the predicted boxes match with the ground truth boxes."""
    matches = []
    unmatched_ground_truths = []
    unmatched_predictions = list(predicted_boxes)
    print("Unmatched Predictions",len(unmatched_predictions))
    for gt_box in ground_truth_boxes:

        max_iou_idx, max_iou = gt_box.find_max_iou_box(unmatched_predictions)
        print("Max IOU",max_iou)
        if max_iou != None and max_iou >= iou_threshold:
            matches.append((gt_box, unmatched_predictions[max_iou_idx]))
            del unmatched_predictions[max_iou_idx]
        else:
            unmatched_ground_truths.append(gt_box)

    return matches, unmatched_ground_truths, unmatched_predictions