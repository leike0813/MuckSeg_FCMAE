from collections.abc import Sequence
import copy
from PIL import Image as PILImage
import numpy as np
import cv2
import torch
import torchvision.transforms.functional as TF


def enlarge_bbox(bbox, bbox_padding):
    return [
        bbox[0] - bbox_padding,
        bbox[1] - bbox_padding,
        bbox[2] + 2 * bbox_padding,
        bbox[3] + 2 * bbox_padding
    ]


def find_mapping_area(global_shape, bbox):
    """
    :param global_shape: H x W
    :param bbox: (left, top, width, height)
    :return: [global_area_width_lower, global_area_width_upper, global_area_height_lower, global_area_height_upper],
             [map_area_width_lower, map_area_width_upper, map_area_height_lower, map_area_height_upper]
    """
    global_width = global_shape[1]
    global_height = global_shape[0]
    global_area_width_lower = bbox[0] if bbox[0] >= 0 else 0
    global_area_height_lower = bbox[1] if bbox[1] >= 0 else 0
    global_area_width_upper = min(bbox[0] + bbox[2] - 1, global_width - 1)
    global_area_height_upper = min(bbox[1] + bbox[3] - 1, global_height - 1)
    map_area_width_lower = 0 if bbox[0] >= 0 else -bbox[0]
    map_area_height_lower = 0 if bbox[1] >= 0 else -bbox[1]
    map_area_width_upper = bbox[2] \
        if bbox[0] + bbox[2] <= global_width - 1 else global_width - 1 - bbox[0]
    map_area_height_upper = bbox[3] \
        if bbox[1] + bbox[3] <= global_height - 1 else global_height - 1 - bbox[1]
    return [global_area_width_lower, global_area_width_upper, global_area_height_lower, global_area_height_upper], \
        [map_area_width_lower, map_area_width_upper, map_area_height_lower, map_area_height_upper]


def cut_image(image, image_shape, bbox, _copy=True):
    if _copy:
        image = copy.deepcopy(image)
    if len(image.shape) == 3:
        _3dim = True
    elif len(image.shape) == 2:
        _3dim = False
    else:
        raise ValueError('Invalid input format')

    if isinstance(image, np.ndarray):
        global_area_geom, map_area_geom = find_mapping_area(image_shape, bbox)
        part_image = image[global_area_geom[2]: global_area_geom[3] + 1,
                           global_area_geom[0]: global_area_geom[1] + 1, :] if _3dim \
            else image[global_area_geom[2]: global_area_geom[3] + 1,
                       global_area_geom[0]: global_area_geom[1] + 1]
    elif isinstance(image, torch.Tensor):
        global_area_geom, map_area_geom = find_mapping_area(image.shape[1:] if _3dim else image.shape, bbox)
        part_image = image[:, global_area_geom[2]: global_area_geom[3] + 1,
                              global_area_geom[0]: global_area_geom[1] + 1] if _3dim \
            else image[global_area_geom[2]: global_area_geom[3] + 1,
                       global_area_geom[0]: global_area_geom[1] + 1]
    else:
        raise ValueError('Invalid input format')
    return part_image


def merge_image(orig_image, new_image, bbox):
    assert orig_image.ndim == new_image.ndim
    _3dim = False
    if orig_image.ndim == 3:
        _3dim = True
        assert orig_image.shape[2] == new_image.shape[2]

    orig_area_geom, new_area_geom = find_mapping_area(orig_image.shape[:2], bbox)
    if _3dim:
        orig_area = orig_image[
                    orig_area_geom[2]: orig_area_geom[3] + 1,
                    orig_area_geom[0]: orig_area_geom[1] + 1,
                    :]
    else:
        orig_area = orig_image[
                    orig_area_geom[2]: orig_area_geom[3] + 1,
                    orig_area_geom[0]: orig_area_geom[1] + 1]
    if _3dim:
        new_area = new_image[
                   new_area_geom[2]: new_area_geom[3] + 1,
                   new_area_geom[0]: new_area_geom[1] + 1,
                   :]
    else:
        new_area = new_image[
                   new_area_geom[2]: new_area_geom[3] + 1,
                   new_area_geom[0]: new_area_geom[1] + 1]
    orig_area += new_area

    return orig_image


def global2part_mask_autoseparate(mask, connectivity=4):
    assert isinstance(mask, np.ndarray) and len(mask.shape) == 2, 'Invalid input format of mask'
    num_con, labeled_instance_mask, stats, center_points = cv2.connectedComponentsWithStats(mask, None, connectivity)
    instance_masks = []
    instance_bboxes = []
    for i in range(num_con - 1):
        instance_mask_global = (labeled_instance_mask == i + 1).astype(np.uint8)
        instance_bbox = stats[i + 1][:4]
        global_area_geom, map_area_geom = find_mapping_area(mask.shape, instance_bbox)
        instance_mask = instance_mask_global[global_area_geom[2]: global_area_geom[3] + 1,
                                             global_area_geom[0]: global_area_geom[1] + 1]
        instance_masks.append(instance_mask)
        instance_bboxes.append(instance_bbox)

    return instance_masks, instance_bboxes


def global2part_image(image, bbox, _copy=True):
    if len(image.shape) == 3:
        _3dim = True
    elif len(image.shape) == 2:
        _3dim = False
    else:
        raise ValueError('Invalid input format')

    if isinstance(image, np.ndarray):
        image_shape = image.shape[:2] if _3dim else image.shape
    elif isinstance(image, torch.Tensor):
        image_shape = image.shape[1:] if _3dim else image.shape
    else:
        raise ValueError('Invalid input format')

    return cut_image(image, image_shape, bbox, _copy)


def part2part_image(image, old_bbox, new_bbox, _copy=True):
    relative_bbox = [new_bbox[0] - old_bbox[0], new_bbox[1] - old_bbox[1], new_bbox[2], new_bbox[3]]
    return cut_image(image, (old_bbox[1], old_bbox[0]), relative_bbox, _copy)


def part2global_image(images, bboxes, global_shape, _clip=None):
    if len(bboxes) == 4 and np.all([isinstance(bboxes[i], (int, float)) for i in range(len(bboxes))]):
        bboxes = [bboxes for i in range(len(images))]
    elif len(bboxes) == len(images) and np.all([isinstance(bboxes[i], Sequence) and len(bboxes[i]) == 4 for i in range(len(bboxes))]):
        pass
    else:
        raise ValueError("Invalid input format of bboxes")

    if np.all([isinstance(image, np.ndarray) for image in images]):
        if np.all([len(image.shape) == 3 for image in images]):
            _3dim = True
            global_image_shape = (global_shape[0], global_shape[1], images.shape[2])
        elif np.all([len(images.shape) == 2 for image in images]):
            _3dim = False
        else:
            raise ValueError('Invalid input format')

        global_image = np.zeros(global_image_shape, dtype=images[0].dtype)
        for image, bbox in zip(images, bboxes):
            global_area_geom, map_area_geom = find_mapping_area(global_shape, bbox)
            global_area = global_image[global_area_geom[2]: global_area_geom[3] + 1,
                                       global_area_geom[0]: global_area_geom[1] + 1, :] if _3dim \
                else global_image[global_area_geom[2]: global_area_geom[3] + 1,
                                  global_area_geom[0]: global_area_geom[1] + 1]
            map_area = image[map_area_geom[2]: map_area_geom[3] + 1,
                            map_area_geom[0]: map_area_geom[1] + 1, :] if _3dim \
                else image[map_area_geom[2]: map_area_geom[3] + 1,
                           map_area_geom[0]: map_area_geom[1] + 1]
            global_area += map_area
        if _clip is not None:
            global_image = np.clip(global_image, *_clip)
    if np.all([isinstance(image, torch.Tensor) for image in images]):
        if np.all([len(image) == 3 for image in images]):
            _3dim = True
            global_image_shape = (images[0].shape[0], global_shape[0], global_shape[1])
        elif np.all([len(images) == 2 for image in images]):
            _3dim = False
        else:
            raise ValueError('Invalid input format')

        global_image = torch.zeros(global_image_shape).astype(images[0].dtype)
        for image, bbox in zip(images, bboxes):
            global_area_geom, map_area_geom = find_mapping_area(global_shape, bbox)
            global_area = global_image[:, global_area_geom[2]: global_area_geom[3] + 1,
                                          global_area_geom[0]: global_area_geom[1] + 1] if _3dim \
                else global_image[global_area_geom[2]: global_area_geom[3] + 1,
                                  global_area_geom[0]: global_area_geom[1] + 1]
            map_area = image[:, map_area_geom[2]: map_area_geom[3] + 1,
                                map_area_geom[0]: map_area_geom[1] + 1] if _3dim \
                else image[map_area_geom[2]: map_area_geom[3] + 1,
                           map_area_geom[0]: map_area_geom[1] + 1]
            global_area += map_area
        if _clip is not None:
            global_image = global_image.clip_(*_clip)

    return global_image


def global2part_contour(contours, bbox=None):
    output_contours = []
    output_bboxes = []
    for contour in contours:
        assert isinstance(contour, np.ndarray) and len(contour.shape) == 3 and contour.shape[1] == 1 and contour.shape[2] == 2, \
            'Invalid input format of contour: {}'.format(contour)

        if bbox is None:
            instance_bbox = cv2.boundingRect(contour)
            output_bboxes.append(instance_bbox)
        else:
            instance_bbox = bbox
        output_contours.append(contour - np.array([[[instance_bbox[0], instance_bbox[1]]]]))

    if bbox is None:
        return output_contours, output_bboxes
    else:
        return output_contours


def part2part_contour(contours, old_bbox, new_bbox):
    relative_bbox = [new_bbox[0] - old_bbox[0], new_bbox[1] - old_bbox[1], new_bbox[2], new_bbox[3]]
    output_contours = []
    for contour in contours:
        output_contours.append(contour - np.array([[[relative_bbox[0], relative_bbox[1]]]]))
    return output_contours


def part2global_contour(contours, bboxes):
    if len(bboxes) == 4 and np.all([isinstance(bboxes[i], (int, float)) for i in range(len(bboxes))]):
        bboxes = [bboxes for i in range(len(contours))]
    elif len(bboxes) == len(contours) and np.all([isinstance(bboxes[i], Sequence) and len(bboxes[i]) == 4 for i in range(len(bboxes))]):
        pass
    else:
        raise ValueError("Invalid input format of bboxes")

    output_contours = []
    for contour, bbox in zip(contours, bboxes):
        output_contours.append(contour + np.array([[[bbox[0], bbox[1]]]]))

    return output_contours


def global2part_center(center, bbox):
    return center[0] - bbox[0], center[1] - bbox[1]


def part2part_center(center, old_bbox, new_bbox):
    relative_bbox = [new_bbox[0] - old_bbox[0], new_bbox[1] - old_bbox[1], new_bbox[2], new_bbox[3]]
    return center[0] - relative_bbox[0], center[1] - relative_bbox[1]


def part2global_center(center, bbox):
    return center[0] + bbox[0], center[1] + bbox[1]


def global2part_bbox(bbox_to_be_convert, bbox):
    converted_anchor = list(global2part_center(bbox_to_be_convert[:2], bbox))
    converted_anchor.extend(bbox_to_be_convert[2:])
    return converted_anchor


def part2part_bbox(bbox_to_be_convert, bbox):
    converted_anchor = list(part2part_center(bbox_to_be_convert[:2], bbox))
    converted_anchor.extend(bbox_to_be_convert[2:])
    return converted_anchor


def part2global_bbox(bbox_to_be_convert, bbox):
    converted_anchor = list(part2global_center(bbox_to_be_convert[:2], bbox))
    converted_anchor.extend(bbox_to_be_convert[2:])
    return converted_anchor