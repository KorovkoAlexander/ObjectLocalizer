import torch


class CustomModel(torch.nn.Module):
    def __init__(self, model):
        super(CustomModel, self).__init__()
        self.base_model = model
        self.fc = torch.nn.Linear(in_features=1000, out_features= 4)

    def forward(self, x):
       x = self.base_model(x)
       x = self.fc(x)
       return x
