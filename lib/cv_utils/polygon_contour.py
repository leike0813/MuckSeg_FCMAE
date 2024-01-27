import numpy as np
import cv2


def polygon_area(points):
    area = 0
    q = points[-1]
    for p in points:
        area += p[0][0] * q[0][1] - p[0][1] * q[0][0]
        q = p
    return area / 2


def filter_max_area_contour(contours):
    max_area_idx = -1
    max_area = 0
    for i, contour in enumerate(contours):
        _area = polygon_area(contour)
        if _area > max_area:
            max_area = _area
            max_area_idx = i
    return [contours[max_area_idx]]


def contour_in_contours_test(contour_to_test, contours_for_test):
    points = [contour_to_test[i, 0].astype(np.float32) for i in range(len(contour_to_test))]
    result = [np.all([cv2.pointPolygonTest(contours_for_test[j], pt, measureDist=False) > 0 for pt in points]) for j in range(len(contours_for_test))]
    return result


def bbox_union(bbox1, bbox2):
    x_min = min(bbox1[0], bbox2[0])
    y_min = min(bbox1[1], bbox2[1])
    return [
        x_min,
        y_min,
        max(bbox1[0] + bbox1[2] - 1, bbox2[0] + bbox2[2] - 1) - x_min + 1,
        max(bbox1[1] + bbox1[3] - 1, bbox2[1] + bbox2[3] - 1) - y_min + 1
    ]


def bbox_overlap(bbox1, bbox2):
    """

    :param bbox1: [x, y, width, height]
    :param bbox2: [x, y, width, height]
    :return:
    """
    if bbox1[0] > bbox2[0] + bbox2[2] - 1:
        return 0.0
    if bbox1[1] > bbox2[1] + bbox2[3] - 1:
        return 0.0
    if bbox1[0] + bbox1[2] - 1 < bbox2[0]:
        return 0.0
    if bbox1[1] + bbox1[3] - 1 < bbox2[1]:
        return 0.0

    x_intersection = min(bbox1[0] + bbox1[2] - 1, bbox2[0] + bbox2[2] - 1) - max(bbox1[0], bbox2[0]) + 1
    y_intersection = min(bbox1[1] + bbox1[3] - 1, bbox2[1] + bbox2[3] - 1) - max(bbox1[1], bbox2[1]) + 1
    area_intersection = x_intersection * y_intersection
    area1 = bbox1[2] * bbox1[3]
    area2 = bbox2[2] * bbox2[3]
    return area_intersection / (area1 + area2 - area_intersection)


def contour_overlap(contour1, bbox1, contour2, bbox2):
    x_min = min(bbox1[0], bbox2[0])
    y_min = min(bbox1[1], bbox2[1])
    x_max = max(bbox1[0] + bbox1[2] - 1, bbox2[0] + bbox2[2] - 1)
    y_max = max(bbox1[1] + bbox1[3] - 1, bbox2[1] + bbox2[3] - 1)
    u_bbox_width = x_max - x_min + 1
    u_bbox_height = y_max - y_min + 1
    u_contour1 = contour1 - np.array([[x_min, y_min]])
    u_contour2 = contour2 - np.array([[x_min, y_min]])
    u_mask_contour1 = np.zeros((u_bbox_height, u_bbox_width), dtype=np.uint8)
    u_mask_contour2 = np.zeros((u_bbox_height, u_bbox_width), dtype=np.uint8)
    u_mask_contour1 = cv2.fillPoly(u_mask_contour1, [u_contour1], 1)
    u_mask_contour2 = cv2.fillPoly(u_mask_contour2, [u_contour2], 1)
    intersection = u_mask_contour1 * u_mask_contour2
    union = u_mask_contour1 + u_mask_contour2 - intersection

    return np.sum(intersection) / np.sum(union)