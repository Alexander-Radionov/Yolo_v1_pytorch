import os
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch

from typing import Tuple, List
from pathlib import Path

from dataclasses import dataclass
import json
from tqdm import tqdm
from math import floor

class_names_mapping = {
    "0": "aeroplane",
    "1": "bicycle",
    "2": "bird",
    "3": "boat",
    "4": "bottle",
    "5": "bus",
    "6": "car",
    "7": "cat",
    "8": "chair",
    "9": "cow",
    "10": "diningtable",
    "11": "dog",
    "12": "horse",
    "13": "motorbike",
    "14": "person",
    "15": "pottedplant",
    "16": "sheep",
    "17": "sofa",
    "18": "train",
    "19": "tvmonitor"
}


@dataclass
class BoundingBox:
    xmin: float
    ymin: float
    xmax: float
    ymax: float

    def __eq__(self, other):
        return self.xmin == other.xmin and self.ymin == other.ymin and self.xmax == other.xmax and self.ymax == other.ymax

    def _calc_intersection(self, other) -> float:
        if self.xmax <= other.xmin or self.xmin >= other.xmax or \
                self.ymax <= other.ymin or self.ymin >= other.ymax:
            return 0.
        if self.xmax > other.xmax:
            x_intersection = self.xmax - other.xmin
        else:
            x_intersection = other.xmax - self.xmin

        if self.ymax > other.ymax:
            y_intersection = self.ymax - other.ymin
        else:
            y_intersection = other.ymax - self.ymin

        intersection = x_intersection * y_intersection
        return intersection

    def _calc_union(self, other, intersection) -> float:
        self_area = (self.xmax - self.xmin) * (self.ymax - self.ymin)
        other_area = (other.xmax - other.xmin) * (other.ymax - other.ymin)
        union = self_area + other_area - intersection
        return union

    def calc_iou(self, other) -> float:
        intersection = self._calc_intersection(other)
        union = self._calc_union(other, intersection)
        return intersection / union

    def calc_iou_with_others(self, others) -> List[float]:
        ious = []
        for other in others:
            ious.append(self.calc_iou(other))
        return ious


def bounding_box_from_yolo_format(x_center, y_center, row_num, col_num, width, height, img_width, img_height,
                                  grid_size) -> BoundingBox:
    x_step = img_width / grid_size
    y_step = img_height / grid_size
    x_center_unsqueezed = row_num * x_step + x_step * x_center
    y_center_unsqueezed = col_num * y_step + y_step * y_center
    xmin = x_center_unsqueezed * img_width - width * img_width * 0.5
    ymin = y_center_unsqueezed * img_height - height * img_height * 0.5
    bbox = BoundingBox(xmin=xmin, ymin=ymin, xmax=xmin + width * img_width, ymax=ymin + height * img_height)
    return bbox


def bounding_box_from_gt_file(x_center, y_center, width, height, img_width, img_height) -> BoundingBox:
    xmin = x_center * img_width - width * img_width * 0.5
    ymin = y_center * img_height - height * img_height * 0.5
    bbox = BoundingBox(xmin=xmin, ymin=ymin, xmax=xmin + width * img_width, ymax=ymin + height * img_height)
    return bbox


class GTobject:
    def __init__(self, class_id, x_center, y_center, width, height, img_width, img_height):
        self.classes_mapping = class_names_mapping
        self.class_id = int(class_id)
        self.class_name = self.classes_mapping[class_id]
        self.img_width = img_width
        self.img_height = img_height

        self.bbox = bounding_box_from_gt_file(x_center, y_center, width, height, img_width, img_height)

        self.width = width
        self.height = height
        self.x_center = x_center
        self.y_center = y_center

    def to_dict(self):
        d = {}
        d['class_id'] = self.class_id
        d['class_name'] = self.class_name
        d['xmin'] = self.bbox.xmin
        d['ymin'] = self.bbox.ymin
        d['xmax'] = self.bbox.xmax
        d['ymax'] = self.bbox.ymax
        d['width'] = self.width
        d['height'] = self.height
        d['img_width'] = self.img_width
        d['img_height'] = self.img_height
        return d


def calc_cell_relative_xy_from_img_relative_xy(x_center, y_center, img_width, img_height, grid_size):
    x_step = img_width / grid_size
    y_step = img_height / grid_size
    row_relative_position = y_center / y_step
    row_num = floor(row_relative_position)
    col_relative_position = x_center / x_step
    col_num = floor(col_relative_position)
    return {'row_num': row_num, 'col_num': col_num}



@dataclass
class DatasetItem:
    id: str
    image: torch.Tensor
    GT_objs: List[GTobject]

    def calc_gt_object_cell_indexes_from_coords(self, bbox, grid_size, x_center, y_center):
        cell_coords = calc_cell_relative_xy_from_img_relative_xy(x_center, y_center, img_width=self.image.shape[2],
                                                                 img_height=self.image.shape[1], grid_size=grid_size)
        center_row_num = cell_coords['row_num']
        center_col_num = cell_coords['col_num']
        x_step = self.image.shape[2] / grid_size
        y_step = self.image.shape[1] / grid_size
        row_relative_position = y_center / y_step
        center_row_num = floor(row_relative_position)
        y_cell_relative_position = row_relative_position - center_row_num
        col_relative_position = x_center / x_step
        center_col_num = floor(col_relative_position)
        x_cell_relative_position = col_relative_position - center_col_num

        min_row_num = int(bbox.ymin // y_step)
        min_col_num = int(bbox.xmin // x_step)
        max_row_num = int(bbox.ymax // y_step)
        max_col_num = int(bbox.xmax // x_step)
        return {'center_row_num': center_row_num,
                'center_col_num': center_col_num,
                'min_row_num': min_row_num,
                'min_col_num': min_col_num,
                'max_row_num': max_row_num,
                'max_col_num': max_col_num,
                'y_cell_relative_position': y_cell_relative_position,
                'x_cell_relative_position': x_cell_relative_position}

    def gt_objs_to_tensor(self, grid_size, predictions_per_cell, num_classes):
        gt_tensor = torch.zeros(size=(grid_size, grid_size, predictions_per_cell * 5 + num_classes))
        for gt_obj in self.GT_objs:
            cell_indexes = self.calc_gt_object_cell_indexes_from_coords(gt_obj.bbox, grid_size=grid_size,
                                                                        x_center=gt_obj.x_center,
                                                                        y_center=gt_obj.y_center)
            class_confidences = [0.] * num_classes
            class_confidences[gt_obj.class_id] = 1.0
            # null_predictions = [0.] * 5 * (predictions_per_cell - 1)  # -1 because there is already one real "prediction"
            gt_item_tensor = torch.FloatTensor(np.array([cell_indexes['x_cell_relative_position'],
                                                         cell_indexes['y_cell_relative_position'],
                                                         gt_obj.width / gt_obj.img_width,
                                                         gt_obj.height / gt_obj.img_height,
                                                         1.0] * predictions_per_cell +  # mb null pred is better
                                                        class_confidences))  # 1.0 for bbox confidence
            # print(f"\nGT item tensor: {gt_item_tensor}")
            # gt_tensor[cell_indexes['min_col_num']:cell_indexes['max_col_num'] + 1,  # +1 to correctly select from tensor
            # cell_indexes['min_row_num']: cell_indexes['max_row_num'] + 1] = gt_item_tensor
            gt_tensor[cell_indexes['center_row_num'], cell_indexes['center_col_num']] = gt_item_tensor

        return gt_tensor


class YoloDataset(Dataset):
    @staticmethod
    def read_gt_items(gt_file_path, img_width, img_height):
        with open(gt_file_path, 'r') as f:
            lines = f.readlines()
        boxes_by_class = []
        for line in lines:
            line_numbers = [float(elem) for elem in line.split(' ')]
            gt_obj = GTobject(class_id=str(int(line_numbers[0])), x_center=line_numbers[1], y_center=line_numbers[2],
                              width=line_numbers[3], height=line_numbers[4], img_width=img_width, img_height=img_height)
            boxes_by_class.append(gt_obj)
        return boxes_by_class

    def _load_item(self, image_file_name, label_file_name, item_id: str) -> DatasetItem:
        img_path = str(Path(self.imgs_dir / image_file_name))
        img = Image.open(img_path)
        img = self.transform(img)

        label_path = str(Path(self.labels_dir) / label_file_name)
        gt_objs = self.read_gt_items(label_path, img_width=img.shape[2], img_height=img.shape[1])
        dataset_item = DatasetItem(id=item_id, image=img, GT_objs=gt_objs)
        return dataset_item

    def __init__(self, path_to_imgs_labels_mapper, grid_size: int, predictions_per_cell: int, num_classes: int,
                 imgs_dir=Path(__file__).parent / 'data' / 'images',
                 labels_dir=Path(__file__).parent / 'data' / 'labels',
                 transform=[transforms.Resize(size=(448, 448)), transforms.ToTensor()], target_transform=None):
        self.imgs_to_labels_mapper = pd.read_csv(path_to_imgs_labels_mapper, sep=',', names=['image_path',
                                                                                             'labels_path'])
        self.imgs_dir = imgs_dir
        self.labels_dir = labels_dir
        self.transform = transforms.Compose(transform)
        self.target_transform = target_transform
        self.dataset_items = []

        self.grid_size = grid_size
        self.predictions_per_cell = predictions_per_cell
        self.num_classes = num_classes

    def __len__(self):
        return len(self.imgs_to_labels_mapper)

    def __getitem__(self, idx):
        data_mapper_row = self.imgs_to_labels_mapper.iloc[idx]
        item_id = data_mapper_row['image_path'].split('.')[0]
        dataset_item = self._load_item(image_file_name=data_mapper_row['image_path'],
                                       label_file_name=data_mapper_row['labels_path'],
                                       item_id=item_id)
        label = dataset_item.gt_objs_to_tensor(grid_size=self.grid_size,
                                               predictions_per_cell=self.predictions_per_cell,
                                               num_classes=self.num_classes)
        return dataset_item.image, label,\
               dataset_item.GT_objs[0].img_width,\
               dataset_item.GT_objs[0].img_height
