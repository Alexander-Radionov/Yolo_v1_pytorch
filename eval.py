import torch
import numpy as np
from model import YoloV1
import torchvision.transforms as transforms
from PIL import Image
from yolo_data_loader import YoloDataset
from torch.utils.data import DataLoader
from yolo_data_loader import class_names_mapping
import torchvision.io as torchio
import matplotlib.pyplot as plt


if __name__ == "__main__":
#     grid_size = 7
#     predictions_per_cell = 2
#     batch_size = 1
#
#     device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")
#     model = YoloV1(grid_size=grid_size, predictions_per_cell=predictions_per_cell, classes=class_names_mapping).to(device)
#     dataset = YoloDataset('data/train.csv', grid_size=grid_size, predictions_per_cell=predictions_per_cell,
#                           num_classes=len(class_names_mapping))
#     train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
#     model.load_state_dict(torch.load('model_checkpoints/model_epoch_0.pkl'))
#     model.eval()
#
#     for i, item in enumerate(train_loader):
#         if i > 0:
#             break
#         input, label, img_width, img_height = item
#         # prediction = model(input.to(device))
#         # print(f"\n X: {input.shape}, y: {label.shape}, prediction: {prediction.shape}, i: {i}")
#         # pred_objs = model.prediction_to_prediction_objects(prediction_from_forward=prediction.to('cpu').squeeze(0),
#         #                                                    img_width=img_width.item(), img_height=img_height.item())
#         # print(f"\nPred objs: {pred_objs}")
#         prediction_objs = model.prediction_to_prediction_objects(prediction_from_forward=label[0], img_width=img_width,
#                                                                  img_height=img_height, grid_size=None)
# print(f"\nObjs: {len(prediction_objs)}")
# for pred_obj in prediction_objs:
#     print(pred_obj)
#         print(f"\nShape: {np.array(input[0]).shape}")
#         plt.imsave('image.jpg', np.moveaxis(np.array(input[0]), 0, 2))
    a = torch.zeros((3, 4))
    a[1, 2] = 1.0
    print(a)