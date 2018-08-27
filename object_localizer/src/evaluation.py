import click
from tqdm import tqdm
from torchvision.models import resnet18
from torchvision import transforms
from torch.utils.data import DataLoader
from .costomized import CustomModel

import os
import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image


all = ["run"]


class EvalDataset(Dataset):
    def __init__(self, dataset_dir, files):
        self.files = files
        self.dataset_dir = dataset_dir
        self.transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor()
        ])
        print("emmiting images from: ", self.dataset_dir)
    
    def __len__(self):
        return len(self.files)

    @classmethod
    def from_file(cls, dataset_dir, filename):
        print("creating eval dataset: ", os.path.join(dataset_dir, filename))
        table = pd.read_csv(os.path.join(dataset_dir, filename))
        dataset_dir = os.path.join(dataset_dir, "images")

        return cls(dataset_dir, list(table.image_name))

    def __getitem__(self, idx):
        path = os.path.join(self.dataset_dir, self.files[idx])
        image = self.transform(Image.open(path))

        return image


def evaluate(loader, model):
    out = []
    for i, images in tqdm(enumerate(loader), total = len(loader)):

        if torch.cuda.is_available():
            images = images.cuda()

        outputs = model(images)/1000
        if torch.cuda.is_available():
            outputs = outputs.cpu()
        out.append(outputs.data.numpy())


    out = np.concatenate(out)
    return out

def run(dataset_path, batch_size = 1):
    files = os.listdir(dataset_path)
    dataset = EvalDataset(dataset_path, files)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)


    model_path = os.path.join(os.path.split(os.path.dirname(__file__))[0], "share", "model_best.pth")
    print("model: ", model_path)
    model = resnet18(pretrained=False)
    model = CustomModel(model)

    if torch.cuda.is_available():
        model.load_state_dict(torch.load(params["model_path"]))
        model.cuda()
    else:
        model.load_state_dict(torch.load(params["model_path"], map_location="cpu"))

    model.eval()

    out = evaluate(loader, model)

    names = dataset.files
    predictions = pd.DataFrame(data=out,
                               index=names,
                               columns=["x1", "y1", "x2", "y2"]
                               )
    predictions.to_csv("predictions.csv")
    print("done")


@click.command()
@click.option("--dataset_path", default="/home/alexander/Downloads/Datasets/AVITO/images_10")
@click.option("--batch_size", default=1)
@click.option("--model_path", default = os.path.join(
    os.path.split(os.path.dirname(__file__))[0],
    "share",
    "model_best.pth")
              )
def main(**kwargs):
    params = {
        "batch_size": kwargs["batch_size"],
        "dataset": kwargs["dataset_path"],
        "model_path":kwargs["model_path"]
    }

    files = os.listdir(params["dataset"])
    dataset = EvalDataset(params["dataset"], files)
    loader = DataLoader(dataset, batch_size=params["batch_size"], shuffle=False)

    model = resnet18(pretrained=False)
    model = CustomModel(model)

    if torch.cuda.is_available():
        model.load_state_dict(torch.load(params["model_path"]))
        model.cuda()
    else:
        model.load_state_dict(torch.load(params["model_path"], map_location="cpu"))
    model.eval()

    out = evaluate(loader, model)

    names = dataset.files
    predictions = pd.DataFrame(data=out,index=names, columns=["x1","y1","x2","y2"])
    predictions.to_csv("predictions.csv")

if __name__ == "__main__":
    main()
