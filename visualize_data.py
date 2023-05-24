from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
import json
from yolo_data_loader import YoloDataset


if __name__ == "__main__":
    dataset = YoloDataset('data/test.csv')
    dataset_iter = iter(dataset)
    dataset_item = next(dataset_iter)
    img = dataset_item.image.numpy()
    img = np.moveaxis(img, [0, 2], [2, 1])
    print(f"\nLoaded image shape: {img.shape}")

    for gt_obj in dataset_item.GT_objs:
        start_point = (int(gt_obj.xmin), int(gt_obj.ymin))
        end_point = (int(gt_obj.xmax), int(gt_obj.ymax))
        color = (0, 255, 0)
        thickness = 2
        img = cv2.rectangle(img, start_point, end_point, color, thickness)
        font = cv2.FONT_HERSHEY_SIMPLEX
        org = (int(gt_obj.xmin), int(gt_obj.ymin-5))
        fontScale = 1
        color = (255, 0, 0)
        thickness = 2
        img = cv2.putText(img, f"{gt_obj.class_name}: {gt_obj.class_id}", org, font,
                            fontScale, color, thickness, cv2.LINE_AA)
    plt.imsave('img.jpg', img)
