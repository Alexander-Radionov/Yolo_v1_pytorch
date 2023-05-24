import torch
import numpy as np
from model import YoloV1, YoloLoss
import torchvision.transforms as transforms
from PIL import Image
from yolo_data_loader import YoloDataset
from torch.utils.data import DataLoader
from yolo_data_loader import class_names_mapping
import torchvision.io as torchio


if __name__ == "__main__":
    grid_size = 7
    predictions_per_cell = 2
    batch_size = 30
    printing_quantity = 10*batch_size
    torch.manual_seed(12334)

    device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")
    model = YoloV1(grid_size=grid_size, predictions_per_cell=predictions_per_cell, classes=class_names_mapping).to(device)
    dataset = YoloDataset('data/train.csv', grid_size=grid_size, predictions_per_cell=predictions_per_cell,
                          num_classes=len(class_names_mapping))
    train_loader = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=2, shuffle=False)

    num_epochs = 2
    # criterion = torch.nn.MSELoss()
    criterion = YoloLoss(grid_size=grid_size, predictions_per_cell=predictions_per_cell, classes=class_names_mapping)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    for epoch_num in range(num_epochs):
        running_loss = 0.
        for i, item in enumerate(train_loader):
            x, y, _, _ = item
            prediction = model(x.to(device))
            optimizer.zero_grad()
            loss = criterion(output=prediction, target=y.to(device), img_width=x.shape[2], img_height=x.shape[3])
            loss.backward()
            optimizer.step()
            running_loss += loss.detach().item()
            if i % printing_quantity == printing_quantity - 1:
                print_loss = round(running_loss / printing_quantity, 3)
                print(f"\nRunning loss at step {i+1} of epoch {epoch_num + 1} is: {print_loss} (total loss: {running_loss})")
                running_loss = 0.
        torch.save(model.state_dict(), f'model_checkpoints/model_epoch_{epoch_num}.pkl')