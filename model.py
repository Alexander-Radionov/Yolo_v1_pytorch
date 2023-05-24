import torch
import torch.nn as nn
from yolo_data_loader import GTobject, bounding_box_from_yolo_format, BoundingBox
from typing import List
import numpy as np
from dataclasses import dataclass


@dataclass
class Prediction:
    bbox: BoundingBox
    class_id: int
    class_name: str
    confidence: float

    def __eq__(self, other):
        attributes_eq = self.class_id == other.class_id and self.class_name == other.class_name and \
                        self.confidence == other.confidence
        boxes_eq = self.bbox == other.bbox
        return attributes_eq and boxes_eq


class YoloV1(nn.Module):
    def __init__(self, grid_size, predictions_per_cell, classes):
        super(YoloV1, self).__init__()
        self.grid_size = grid_size
        self.predictions_per_cell = predictions_per_cell
        self.classes_mapping = classes

        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.1)
        self.conv1_1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3)

        self.conv2_1 = nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3, padding=1)

        self.conv3_1 = nn.Conv2d(in_channels=192, out_channels=128, kernel_size=1)
        self.conv3_2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1)
        self.conv3_4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)

        self.conv4_1 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1)
        self.conv4_2 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1)
        self.conv4_4 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=1)

        self.conv5_1 = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1)
        self.conv5_2 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, padding=1)
        self.conv5_4 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=2, padding=1)

        self.conv6_1 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, padding=1)

        self.connected_flat = nn.Linear(in_features=1024*self.grid_size * self.grid_size, out_features=496)
        self.connected_output = nn.Linear(in_features=496,
                                          out_features=self.grid_size * self.grid_size * (self.predictions_per_cell*5 + len(self.classes_mapping)))

    def forward(self, x):
        # print(f"Init: {x.shape}")
        x = self.max_pool(self.leaky_relu(self.conv1_1(x)))
        # print(f"1: {x.shape}")
        x = self.max_pool(self.leaky_relu(self.conv2_1(x)))
        # print(f"2: {x.shape}")
        x = self.leaky_relu(self.conv3_1(x))
        # print(f"3_1: {x.shape}")
        x = self.leaky_relu(self.conv3_2(x))
        # print(f"3_2: {x.shape}")
        x = self.leaky_relu(self.conv3_3(x))
        # print(f"3_3: {x.shape}")
        x = self.max_pool(self.leaky_relu(self.conv3_4(x)))
        # print(f"3: {x.shape}")
        x = self.leaky_relu(self.conv4_1(x))
        # print(f"4_1: {x.shape}")
        x = self.leaky_relu(self.conv4_2(x))
        # print(f"4_2: {x.shape}")
        x = self.leaky_relu(self.conv4_1(x))
        # print(f"4_3: {x.shape}")
        x = self.leaky_relu(self.conv4_2(x))
        # print(f"4_4: {x.shape}")
        x = self.leaky_relu(self.conv4_3(x))
        # print(f"4_5: {x.shape}")
        x = self.max_pool(self.leaky_relu(self.conv4_4(x)))
        # print(f"4: {x.shape}")
        x = self.leaky_relu(self.conv5_1(x))
        # print(f"5_1: {x.shape}")
        x = self.leaky_relu(self.conv5_2(x))
        # print(f"5_2: {x.shape}")
        x = self.leaky_relu(self.conv5_1(x))
        # print(f"5_3: {x.shape}")
        x = self.leaky_relu(self.conv5_2(x))
        # print(f"5_4: {x.shape}")
        x = self.leaky_relu(self.conv5_3(x))
        # print(f"5_5: {x.shape}")
        x = self.leaky_relu(self.conv5_4(x))
        # print(f"5: {x.shape}")
        x = self.leaky_relu(self.conv6_1(x))
        # print(f"6_1: {x.shape}")
        x = self.leaky_relu(self.conv6_1(x))
        # print(f"6_2: {x.shape}")
        flat_1 = nn.Flatten(start_dim=1)
        x = flat_1(x)
        x = self.leaky_relu(self.connected_flat(x))
        # x = torch.sum(x,)
        # print(x.shape)
        # relu = torch.nn.ReLU()
        x = torch.clip(self.connected_output(x), 0., 1.)  # TODO: check if constrain is really needed. Paper says differntly
        # print(x.shape)
        x = torch.reshape(x, (-1, self.grid_size, self.grid_size,
                              self.predictions_per_cell * 5 + len(self.classes_mapping)))

        return x

    def choose_prediction_for_single_cell(self, cell_num, cells, prediction_from_forward):
        row_ind = cell_num // self.grid_size
        col_ind = cell_num % self.grid_size
        pred_vector = prediction_from_forward[row_ind, col_ind].tolist()
        class_confidences = pred_vector[:self.predictions_per_cell * 5]
        max_box_confidence = 0.
        for i in range(self.predictions_per_cell):
            box_confidence = pred_vector[i + 4]
            if box_confidence > max_box_confidence:
                max_box_confidence = box_confidence
                cur_coords = {'obj_x_center': pred_vector[i], 'obj_y_center': pred_vector[i + 1],
                              'obj_width': pred_vector[i + 2],
                              'obj_height': pred_vector[i + 3],
                              'box_confidence': box_confidence}

                class_box_confidences = [conf * box_confidence for conf in class_confidences]
                # if min(class_box_confidences) >= min_confidence_threshold:
                cells[cell_num].update(cur_coords)
                cells[cell_num]['class_confidences'] = class_box_confidences
                cells[cell_num]['predicted_class_id'] = np.argmax(class_box_confidences)
            i += 4

    def collect_class_predictions_accross_cells(self, cells, class_id, img_height, img_width, grid_size,
                                                min_confidence) -> List[Prediction]:
        class_predictions = []
        for cell_num in cells:
            cell = cells[cell_num]
            if len(cell.keys()) > 0 and cell['predicted_class_id'] == class_id and \
                    cell['box_confidence'] > min_confidence:
                pred_box = bounding_box_from_yolo_format(x_center=float(cell['obj_x_center']),
                                                         y_center=float(cell['obj_y_center']), row_num=None,
                                                         col_num=None, width=float(cell['obj_width']),
                                                         height=float(cell['obj_height']), img_width=float(img_width),
                                                         img_height=float(img_height), grid_size=None)
                prediction = Prediction(bbox=pred_box, class_id=cell['predicted_class_id'],
                                        class_name='Unknown', confidence=cell['box_confidence'])
                if prediction not in class_predictions:
                    class_predictions.append(prediction)
        return class_predictions

    def filter_class_predictions(self, class_predictions: List[Prediction], min_iou=0.5) -> List[Prediction]:
        chosen_class_predictions = {i: {'chosen': True,
                                        'prediction': pred,
                                        'compared_to': set([i])} for i,
                                                                     pred in enumerate(class_predictions)}
        for class_pred_id, class_pred in chosen_class_predictions.items():
            other_boxes = []
            for pred in chosen_class_predictions.values():
                other_box = pred['prediction'].bbox
                other_boxes.append(other_box)
            pred_ious = class_pred['prediction'].bbox.calc_iou_with_others(other_boxes)
            for intersected_pred_id, iou in enumerate(pred_ious):
                if iou < min_iou or class_pred_id == intersected_pred_id or \
                        class_pred_id in chosen_class_predictions[intersected_pred_id]['compared_to']:  # don't compare non-intersecting preds or prediction with itself or with items already compared
                    continue
                else:
                    if chosen_class_predictions[intersected_pred_id]['prediction'].confidence > class_pred['prediction'].confidence:
                        chosen_class_predictions[intersected_pred_id]['chosen'] = True
                        chosen_class_predictions[class_pred_id]['chosen'] = False
                    else:
                        chosen_class_predictions[class_pred_id]['chosen'] = True
                        chosen_class_predictions[intersected_pred_id]['chosen'] = False
                    chosen_class_predictions[intersected_pred_id]['compared_to'].add(class_pred_id)
                    chosen_class_predictions[class_pred_id]['compared_to'].add(intersected_pred_id)
        return [elem['prediction'] for elem in chosen_class_predictions.values() if elem['chosen']]

    def prediction_to_prediction_objects(self, prediction_from_forward: torch.Tensor, img_width: int, img_height: int,
                                         grid_size: int, min_confidence_threshold=0.0) -> List[Prediction]:
        cells = {i: {} for i in range(self.grid_size * self.grid_size)}
        for cell_num in range(len(cells)):
            self.choose_prediction_for_single_cell(cell_num, cells, prediction_from_forward)
        chosen_predictions = []
        for class_id in range(len(self.classes_mapping)):
            class_predictions = self.collect_class_predictions_accross_cells(cells, class_id, img_height, img_width,
                                                                             grid_size, min_confidence_threshold)
            if len(class_predictions) > 0:
                chosen_class_predictions = self.filter_class_predictions(class_predictions)
                chosen_predictions.extend(chosen_class_predictions)
        return chosen_predictions


class YoloLoss(nn.Module):
    def __init__(self, grid_size, predictions_per_cell, classes):
        super(YoloLoss, self).__init__()
        self.grid_size = grid_size
        self.predictions_per_cell = predictions_per_cell
        self.classes_mapping = classes
        self.mse = nn.MSELoss(reduction="sum")

    def calc_target_cell_values(self, col_ind, img_height, img_width, row_ind, target):
        predictor_ind = 0
        target_vector = target[:, row_ind, col_ind]
        target_coords = {'x_center': target_vector[..., predictor_ind],
                         'y_center': target_vector[..., predictor_ind + 1],
                         'width': target_vector[..., predictor_ind + 2] * img_width,
                         'height': target_vector[..., predictor_ind + 3] * img_height}
        target_box = bounding_box_from_yolo_format(**target_coords,
                                                   row_num=row_ind, col_num=col_ind,
                                                   img_width=img_width, img_height=img_height,
                                                   grid_size=self.grid_size)
        target_class_confidences = target_vector[..., self.predictions_per_cell * 5:]
        item_in_cell = target_vector[..., 4]
        target_box_confidence = target_vector[..., 5 * predictor_ind + 4]
        target = {'coords': target_coords, 'box': target_box, 'vector': target_vector,
                  'box_confidence': target_box_confidence,
                  'class_confidences': target_class_confidences, 'has_item': item_in_cell}
        return target

    def calc_prediction_cell_values(self, output, row_ind, col_ind, img_width, img_height, target_box):
        pred_vector = output[:, row_ind, col_ind]
        class_confidences = pred_vector[..., self.predictions_per_cell * 5:]
        max_iou_with_gt = 0.
        pred_box = BoundingBox(xmin=0., ymin=0., xmax=0., ymax=0.)
        pred_box_confidence = 0.
        class_box_confidences = torch.zeros(size=(1, len(self.classes_mapping)), device=output.device)
        cur_coords = {key: 0. for key in ['x_center', 'y_center', 'width', 'height']}
        for predictor_ind in range(self.predictions_per_cell):
            pred_box_confidence = pred_vector[..., 5 * predictor_ind + 4]
            pred_width = pred_vector[..., 5 * predictor_ind + 3] * img_width
            pred_height = pred_vector[..., 5 * predictor_ind + 4] * img_height
            cur_coords = {'x_center': pred_vector[..., predictor_ind * 5],
                          'y_center': pred_vector[..., predictor_ind * 5 + 1],
                          'width': pred_width,
                          'height': pred_height}
            # TODO: check
            pred_box = bounding_box_from_yolo_format(**cur_coords,
                                                     row_num=row_ind, col_num=col_ind,
                                                     img_width=img_width, img_height=img_height,
                                                     grid_size=self.grid_size)
            iou_with_gt = pred_box.calc_iou(target_box)
            if iou_with_gt >= max_iou_with_gt:  # >= to train all predictors in case there is no obj
                max_iou_with_gt = iou_with_gt
                cur_coords['box_confidence'] = pred_box_confidence
                class_box_confidences = class_confidences * pred_box_confidence
        prediction = {'coords': cur_coords, 'box': pred_box, 'vector': pred_vector,
                      'box_confidence': pred_box_confidence,
                      'class_confidences': class_box_confidences}
        return prediction

    def forward(self, output, target, img_width, img_height, lambda_coord=5.0, lambda_nobj=0.5) -> torch.FloatTensor:
        loss = torch.zeros(size=(1, 1), device=output.device)
        for cell_ind in range(self.grid_size * self.grid_size):
            xy_loss = torch.zeros(size=(1, 1), device=output.device)
            wh_loss = torch.zeros(size=(1, 1), device=output.device)
            box_confidences_loss = torch.zeros(size=(1, 1), device=output.device)
            no_obj_box_confidences_loss = torch.zeros(size=(1, 1), device=output.device)
            class_confidences_loss = torch.zeros(size=(1, 1), device=output.device)

            row_ind = cell_ind // self.grid_size
            col_ind = cell_ind % self.grid_size
            target_values = self.calc_target_cell_values(col_ind, img_height, img_width, row_ind, target)
            prediction_values = self.calc_prediction_cell_values(output, row_ind, col_ind, img_width, img_height,
                                                                 target_values['box'])
            x_loss = self.mse(target_values['coords']['x_center'], prediction_values['coords']['x_center'])
            y_loss = self.mse(target_values['coords']['y_center'], prediction_values['coords']['y_center'])
            xy_loss += lambda_coord * target_values['has_item'] * (x_loss + y_loss)

            w_loss = self.mse(torch.sqrt(target_values['coords']['width']),
                              torch.sqrt(prediction_values['coords']['width']))
            h_loss = self.mse(torch.sqrt(target_values['coords']['height']),
                              torch.sqrt(prediction_values['coords']['height']))
            wh_loss += lambda_coord * target_values['has_item'] * (w_loss + h_loss)

            box_confidences_loss += target_values['has_item'] * self.mse(target_values['box_confidence'],
                                                                         prediction_values['box_confidence'])
            no_obj_box_confidences_loss += lambda_nobj * (1 - target_values['has_item']) * self.mse(target_values['box_confidence'],
                                                    prediction_values['box_confidence'])
            class_confidences_loss += self.mse(target_values['class_confidences'],
                                               prediction_values['class_confidences'])
            cell_loss = xy_loss + wh_loss + box_confidences_loss + no_obj_box_confidences_loss + class_confidences_loss
            loss += cell_loss
            # print(f"\nLoss on cell {cell_ind} is {loss.detach().item()}")
        return loss
