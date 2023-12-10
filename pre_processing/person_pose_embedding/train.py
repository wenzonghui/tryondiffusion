from network import AutoEncoder
from utils.dataloader import KeypointDataset

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import os
from torch.utils.tensorboard import SummaryWriter


def train(train_dir,
          test_dir,
          num_epochs=100,
          save_model_dir='./models'):
    writer = SummaryWriter("")

    if not os.path.exists(save_model_dir):
        os.mkdir(save_model_dir)

    train_data = KeypointDataset(train_dir)
    train_dataloader = DataLoader(train_data, batch_size=16, shuffle=True)

    test_data = KeypointDataset(test_dir)
    test_dataloader = DataLoader(test_data, batch_size=4)

    model = AutoEncoder(34)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    best_loss = np.inf

    for epoch in range(num_epochs):
        train_running_loss = 0
        num_batches = 0
        for keypoints, _ in train_dataloader:

            model.train()
            prediction, _ = model(keypoints)
            loss = criterion(prediction, keypoints)
            writer.add_scalar("Running_Loss/Train", loss, epoch + 1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_running_loss += float(loss)
            num_batches += 1

            if not num_batches % 125:
                print(f"Running Loss; Iteration: {num_batches}, Train Loss: {(train_running_loss / num_batches):.4f}")

        print(f"Epoch: {epoch + 1}/{num_epochs}, Train Loss: {(train_running_loss / num_batches):.4f}\n")
        writer.add_scalar("Epoch_Loss/Train", train_running_loss / num_batches, epoch + 1)

        # validation
        test_running_loss = 0
        num_test_batches = 0

        for test_keypoints, _ in test_dataloader:
            model.eval()
            test_predictions, _ = model(test_keypoints)
            test_loss = criterion(test_predictions, test_keypoints)
            writer.add_scalar("Running_Loss/Validation", test_loss, epoch + 1)

            test_running_loss += float(test_loss)
            num_test_batches += 1

        print(f"Epoch: {epoch + 1}/{num_epochs}, Test Loss: {(test_running_loss / num_test_batches):.4f}")
        writer.add_scalar("Epoch_Loss/Validation", test_running_loss / num_test_batches, epoch + 1)

        if test_running_loss / num_test_batches < best_loss:
            print(f"Best Model Till Now: {epoch + 1}")

            best_loss = test_running_loss / num_test_batches
            torch.save(model.state_dict(), os.path.join(save_model_dir, f"fc1.pth"))

        # 保存每一个模型
        # torch.save(model.state_dict(), os.path.join(save_model_dir, f"{epoch + 1}.pth"))

    writer.flush()
    writer.close()


if __name__ == "__main__":
    train(train_dir="/home/xkmb/下载/data/train/jp",
          test_dir="/home/xkmb/下载/data/val/jp",
          save_model_dir="/home/xkmb/下载/tryondiffusion/models",
          num_epochs=300)
