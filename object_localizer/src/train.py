import os

import click
import numpy as np

import torch
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torchvision.models import resnet18
import torch.optim as optim

from .dataset import PascalDataset
from .costomized import CustomModel

def compute_iou(box1, box2):
    """
    Returns the IoU of two bounding boxes
    """

    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)

    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * \
                 torch.clamp(inter_rect_y2 - inter_rect_y1 + 1, min=0)


    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou


class Trainer():
    def __init__(self, train_loader, val_loader, params):

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.params = params
        self.logs_dir = os.path.join(params["save_dir"], "logs")
        self.runs_dir = os.path.join(params["save_dir"], "runs")
        self.checkpoint_dir = os.path.join(params["save_dir"], "checkpoints")

        os.makedirs(self.logs_dir, exist_ok=True)
        os.makedirs(os.path.join(self.logs_dir, "train"), exist_ok=True)
        os.makedirs(os.path.join(self.logs_dir, "val"), exist_ok=True)
        os.makedirs(os.path.join(params["save_dir"], "runs"), exist_ok=True)
        os.makedirs(os.path.join(params["save_dir"], "checkpoints"), exist_ok=True)

        self.train_writer = SummaryWriter(log_dir=os.path.join(self.logs_dir, "train"))
        self.val_writer = SummaryWriter(log_dir=os.path.join(self.logs_dir, "val"))

    @staticmethod
    def run(model, optimizer, loader, criterion, train_writer, params, steps, mode = "train"):
        running_loss = []
        running_iou = []
        total_loss = []
        total_iou = []
        current_steps = steps
        for i, (images, targets) in enumerate(loader):
            if params["cuda"]:
                images = images.cuda()
                targets = targets.cuda()

            if mode == "train":
                model.train()
            else:
                model.eval()

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            iou = compute_iou(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss.append(loss.item())
            running_iou.append(iou.mean().item())
            total_loss.append(loss.item())
            total_iou.append(iou.mean().item())
            current_steps += 1

            if current_steps % params["show_intv"] == params["show_intv"] - 1:
                train_writer.add_scalar(f"loss", np.mean(running_loss), current_steps)
                print(f"step {i} current loss: ", np.mean(running_loss))
                train_writer.add_scalar(f"iou", np.mean(running_iou), current_steps)
                print(f"step {i} current iou: ", np.mean(running_iou))
                running_loss = []
                running_iou = []

        return np.mean(total_loss), current_steps

    def fit(self, model, optimizer, criterion, scheduler):
        best_loss = 999999
        steps_train = 0
        steps_valid = 0


        for epoch in range(self.params["epochs"]):
            print(f"training epoch {epoch + 1}")
            loss_train, steps = self.run(model, optimizer, self.train_loader, criterion, self.train_writer,
                                         self.params,
                                         steps_train, mode="train")
            steps_train = steps

            print(f"validating epoch {epoch + 1}")
            loss_val, steps = self.run(model, optimizer, self.val_loader, criterion, self.val_writer, self.params,
                                       steps_valid, mode="val")
            steps_valid = steps

            if loss_val < best_loss:
                best_loss = loss_val
                torch.save(model.state_dict(), os.path.join(self.checkpoint_dir, f"model_best.pth"))
            # torch.save(model.state_dict(), os.path.join(self.checkpoint_dir, f"model_epoch_{epoch}.pth"))
            # scheduler.step()


@click.command()
@click.argument("save_dir")
@click.option("--batch_size", default=100)
@click.option("--show_interval", default=1)
@click.option("--dataset_path", default="/home/alexander/Downloads/Datasets/AVITO")
@click.option("--epochs", default=500)
def main(**kwargs):
    params = {
        "save_dir": kwargs["save_dir"],
        "batch_size": kwargs["batch_size"],
        "show_intv": kwargs["show_interval"],
        "cuda": torch.cuda.is_available(),
        "dataset": kwargs["dataset_path"],
        "epochs": kwargs["epochs"]
    }

    dataset_train = PascalDataset.from_file(kwargs["dataset_path"],"train_5491.csv")
    dataset_val = PascalDataset.from_file(kwargs["dataset_path"],"val_5491.csv")
    train_loader = DataLoader(dataset_train, batch_size=params["batch_size"], shuffle = True)
    val_loader = DataLoader(dataset_val, batch_size=params["batch_size"], shuffle = True)

    model = resnet18(pretrained=False)
    model = CustomModel(model)

    if params["cuda"]:
        model.cuda()

    print("model created")
    criterion = torch.nn.SmoothL1Loss(size_average=False)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 200, 300])

    trainer = Trainer(train_loader, val_loader, params)
    print("fitting!..")
    trainer.fit(model, optimizer, criterion, scheduler)


if __name__ == "__main__":
    main()












