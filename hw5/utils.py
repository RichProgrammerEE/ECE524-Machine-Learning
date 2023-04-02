import os
from pathlib import Path
from typing import List, Callable, Optional

import pandas as pd
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as Fun
from torch.utils.data import Dataset
from PIL import Image
from tqdm import tqdm


def data_dir() -> Path:
    return Path(__file__).parent / "data"


def count_trainable_parameters(model: torch.nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class FeatureModel(torch.nn.Module):
    """Takes a pretrained model, extracts the output from the specified layer an
    wraps it in a torch.nn.Module"""

    def __init__(self, model: torch.nn.Module, layer_name: str, **kwargs) -> None:
        super().__init__(**kwargs)

        # Use this to print available layer names
        # print(get_graph_node_names(model))

        # Create feature extraction wrapper that pull output from specified layer
        self.feature_model = create_feature_extractor(
            model, return_nodes={layer_name: "feature_output"})

    def forward(self, x):
        x = self.feature_model(x)
        return x["feature_output"]


class FaceDataset(Dataset):
    """Pytorch wrapper around our face dataset"""

    def __init__(self,
                 csv_file: Path,
                 root_dir: Path,
                 transform: Optional[Callable] = None) -> None:
        super().__init__()

        self.df = pd.read_csv(csv_file, index_col=0)
        self.categories = pd.Categorical(self.df["Category"])
        self.df["cat"] = self.categories.codes
        self.root_dir = root_dir
        self.transform = transform

    def num_categories(self) -> int:
        return self.categories.categories.size

    def get_label_desc(self, label: int) -> str:
        return self.categories.categories[label]

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        # From the index, look up the data
        data = self.df.iloc[idx].to_list()
        image_path = str(self.root_dir / data[0])
        sample = Image.open(image_path).convert("RGB")

        if self.transform:
            sample = self.transform(sample)

        category: int = data[2]
        # print(category)
        # labels = Fun.one_hot(torch.tensor(
        #     [category]).long(), num_classes=self.num_categories())
        return sample, torch.tensor(category).to(torch.int64)


class Trainer():
    def __init__(self, num_classes, epochs) -> None:
        self.running_loss = 0.0
        self.print_loss_interval = num_classes
        self.print_loss_count = 0
        self.total_sample_count = 0
        self.epochs = epochs

    def __print_stats(self, epoch):
        print(
            f'[{epoch}, {self.total_sample_count:5d}] loss: {self.running_loss / self.print_loss_count:.3f}')
        self.running_loss = 0.0
        self.print_loss_count = 0

    def train(self,
              data_loader: DataLoader,
              feature_model: torch.nn.Module,
              #   classifier,
              criterion,
              optimizer,
              scheduler):
        # Set the classifier for training
        print("Training...")
        # classifier.train(mode=True)
        total_samples = self.epochs * len(data_loader) * data_loader.batch_size
        with tqdm(total=total_samples) as progress:
            for epoch in range(self.epochs):  # Loop over the dataset multiple times
                for _, data in enumerate(data_loader):
                    # Get the inputs
                    inputs, labels = data
                    # feature_outputs = feature_model.forward_features(inputs)

                    # Number of samples in batch
                    batch_count = inputs.size()[0]

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward + backward + optimize
                    # m = torch.nn.LogSoftmax(dim=1)  # For NLL Loss
                    outputs = feature_model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    # scheduler.step()

                    # for param in classifier.parameters():
                    #     print(param)
                    #     print(param.grad)

                    # print statistics
                    self.running_loss += loss.item() * batch_count
                    self.print_loss_count += batch_count
                    self.total_sample_count += batch_count
                    if self.print_loss_count >= self.print_loss_interval:
                        self.__print_stats(epoch + 1)
                    progress.update(batch_count)

        self.__print_stats("Final")
        print('Finished Training')
