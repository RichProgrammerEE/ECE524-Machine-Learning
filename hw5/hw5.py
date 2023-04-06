import os
import time
import csv
import warnings
import argparse
from pathlib import Path
from typing import List, Callable, Optional

# SEE: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
# SEE: https://towardsdatascience.com/finetune-a-facial-recognition-classifier-to-recognize-your-face-using-pytorch-d00a639d9a79
import torch
from torch.utils.data import random_split, DataLoader, Dataset
import torch.nn.functional as Fun
import torchvision
from torchvision import transforms
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names
from torchsummary import summary
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
# SEE: https://github.com/huggingface/pytorch-image-models/tree/7501972cd61dde7428164041b0a6dd8fea60c4d4/timm/models
from timm.models import InceptionV4, InceptionResnetV2
from PIL import Image
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt


def data_dir() -> Path:
    return Path(__file__).parent / "data"


def output_dir() -> Path:
    return Path(__file__).parent / "outputs"


def get_torch_device() -> torch.device:
    # return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def count_trainable_parameters(model: torch.nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class FaceTrainDataset(Dataset):
    """Pytorch wrapper around our face testing dataset"""

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

    def reset_transform(self, transform: transforms.Compose):
        self.transform = transform

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        # From the index, look up the data
        data = self.df.iloc[idx].to_list()
        image_path = str(self.root_dir / data[0])
        # Ignore warning about transparency channel
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            sample = Image.open(image_path).convert("RGB")

        if self.transform:
            sample = self.transform(sample)

        category: int = data[2]
        # Return the Image, Category Integer, Category name
        return sample, torch.tensor(category).to(torch.int64), data[1]


class FaceTestDataset(Dataset):
    """Pytorch wrapper around our face training dataset"""

    def __init__(self,
                 root_dir: Path,
                 transform: Optional[Callable] = None) -> None:
        super().__init__()

        # Need to list all the files in the test directory
        self.test_files = os.listdir(root_dir)
        self.test_files.sort(key=lambda x: int(Path(x).stem))
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.test_files)

    def __getitem__(self, idx):
        # From the index, look up the data
        image_path = str(self.root_dir / self.test_files[idx])
        # Ignore warning about transparency channel
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            sample = Image.open(image_path).convert("RGB")

        if self.transform:
            sample = self.transform(sample)

        # Return the Image, Index
        return sample, idx


class Trainer():
    def __init__(self, num_classes, epochs) -> None:
        self.running_loss = 0.0
        self.print_loss_interval = num_classes * 20
        self.print_loss_count = 0
        self.total_sample_count = 0
        self.epochs = epochs

    def __print_stats(self, epoch):
        # print(
        #     f'[{epoch}, {self.total_sample_count:5d}] loss: {self.running_loss / self.print_loss_count:.3f}')
        self.running_loss = 0.0
        self.print_loss_count = 0

    def train(self,
              data_loader: DataLoader,
              model: torch.nn.Module,
              criterion: torch.nn.CrossEntropyLoss,
              optimizer: torch.optim.SGD,
              scheduler: torch.optim.lr_scheduler.StepLR) -> List[float]:
        # Set the classifier for training
        print("Training...")
        losses = []
        start = time.time()
        device = get_torch_device()
        # Set model to training mode
        model.train()
        model = model.to(device)
        total_samples = self.epochs * len(data_loader) * data_loader.batch_size
        with tqdm(total=total_samples) as progress:
            for epoch in range(self.epochs):  # Loop over the dataset multiple times
                for _, data in enumerate(data_loader):
                    # Get the inputs
                    inputs, labels, _ = data
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # Number of samples in batch
                    batch_count = inputs.size()[0]

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward + backward + optimize
                    # m = torch.nn.LogSoftmax(dim=1)  # For NLL Loss
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    # for param in classifier.parameters():
                    #     print(param)
                    #     print(param.grad)

                    # Track loss
                    losses.append(loss.item())

                    # print statistics
                    self.running_loss += loss.item() * batch_count
                    self.print_loss_count += batch_count
                    self.total_sample_count += batch_count
                    if self.print_loss_count >= self.print_loss_interval:
                        self.__print_stats(epoch + 1)
                    progress.update(batch_count)
                # Step the scheduler after each step (decay the optimizer)
                scheduler.step()

        self.__print_stats("Final")
        time_elapsed = time.time() - start
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        return losses


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", help="train the model",
                        action="store_true")
    parser.add_argument("-v", "--validate", help="validate the model on testing split of data",
                        action="store_true")
    parser.add_argument("--test", help="test the model",
                        action="store_true")
    parser.add_argument("-d", "--dataset", type=str, default="train_small",
                        help="train or train_small")
    parser.add_argument("-w", "--weights", type=str,
                        help="weights to use for validation and testing phase")
    args = parser.parse_args()

    # Define device for pytorch
    device = get_torch_device()
    print(f"Using device: {str(device).upper()}")
    # Random seed
    seed = torch.Generator().manual_seed(43)

    os.makedirs(str(output_dir()), exist_ok=True)

    ############################## PRE-TRAINED MODEL SETUP ##############################
    # Basic idea here is to take a pretrained classification model and remove the last
    # few layers (e.g. flatten, pooling, softmax). Then re-attach the last few layers
    # with "requires_grad" = True and retrain the last few layers using our training data.

    # List all available timm pretrained models
    print(timm.list_models(pretrained=True))

    # SEE: https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/inception_v4.py
    model: InceptionResnetV2 = timm.create_model(
        "efficientnet_b4", pretrained=True)
    # Set model to eval mode for inference
    model.eval()
    # Freeze gradient calculation (we will not be training this model)
    for param in model.parameters():
        param.requires_grad = False

    # Setup data transformation for input
    config = resolve_data_config({}, model=model)
    print(config)
    transform = create_transform(**config)
    print(transform)
    # Custom transform for more robust training
    # training_transform = transforms.Compose([
    #     transforms.Resize(size=(256, 256), interpolation=transforms.InterpolationMode.BICUBIC, antialias="warn"),
    #     transforms.CenterCrop(size=(224, 224)),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ColorJitter(),
    #     transforms.GaussianBlur(kernel_size=(9, 9), sigma=(0.1, 5)),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=torch.tensor([0.4850, 0.4560, 0.4060]), std=torch.tensor([0.2290, 0.2240, 0.2250]))
    # ])
    training_transform = transform

    ############################## LOAD DATASETS ##############################

    dataset_string = args.dataset
    dataset = FaceTrainDataset(
        csv_file=data_dir() / f"{dataset_string}.csv",
        root_dir=data_dir() / f"{dataset_string}",
        transform=training_transform
    )

    # Split the training data into train + test datasets
    train_dataset, val_dataset = random_split(
        dataset, [0.95, 0.05], generator=seed)
    print("Dataset Split - Train: {}, Test: {}, Total: {}".format(
        len(train_dataset), len(val_dataset), len(dataset)
    ))
    print(f"Number of classes in dataset: {dataset.num_categories()}")

    batch_size = 4
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    ############################## SHOW DATASET SAMPLE ##############################

    # # Show a few images
    # train_iter = iter(train_loader)
    # images, labels, _ = next(train_iter)

    # for data in list(zip(images, labels)):
    #     image = data[0]
    #     label = data[1].item()
    #     print(f"Category: {label}, Label: {dataset.get_label_desc(label)}")
    #     torchvision.transforms.ToPILImage()(image).show()
    #     tensor = data[0].unsqueeze(0)
    #     print(model.forward_features(tensor).size())

    ############################## RESET THE MODEL CLASSIFIER ##############################

    model.reset_classifier(num_classes=100)
    print(f"Trainable model parameters: {count_trainable_parameters(model)}")

    # Print information about the model
    # print(model)
    # summary(model, (3, 299, 299), device=str(device))

    ############################## CREATE LOSS + OPTIMIZER ##############################

    # Define a loss function and an optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.1, momentum=0.9)
    # Decay LR by a factor of *gamma* every *step_size* epochs
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=4, gamma=0.1)

    ############################## TRAIN THE CLASSIFIER ##############################

    model_weights_path: Path = output_dir() / args.weights
    num_epochs = 20
    if args.train:
        trainer = Trainer(dataset.num_categories(), num_epochs)
        training_loss = trainer.train(train_loader, model,
                                      criterion, optimizer, scheduler)

        # Save the model weights
        torch.save(model.state_dict(), model_weights_path)

    ############################## VALIDATE THE CLASSIFIER ##############################

    if args.validate:
        print("Validating...")
        model.load_state_dict(torch.load(model_weights_path))
        model = model.to(device)
        # Reset the dataset transform for validation
        dataset.reset_transform(transform)

        with torch.no_grad(), tqdm(total=len(val_dataset)) as progress:
            # Evaluate the model on our test split
            total = 0
            correct = 0
            for data in val_loader:
                # Calculate outputs by running images through the network
                inputs, labels, names = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)

                batch_count = labels.size(0)
                # The class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs.data, 1)
                total += batch_count
                correct += (predicted == labels).sum().item()
                progress.update(batch_count)

                # name = names[0]
                # pred = predicted.cpu().tolist()[0]
                # print(predicted == labels, name, pred, dataset.get_label_desc(pred))
        print(f'Accuracy on the {total} test images: {100 * correct // total} %')

    ############################## TEST THE CLASSIFIER ##############################
    # Uses the trained model to test the classifier on the testing data and generate
    # the appropriate kaggle submission csv

    if args.test:
        test_dataset = FaceTestDataset(
            root_dir=data_dir() / "test",
            transform=transform
        )

        batch_size = 4
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        print(f"Test Dataset: {len(test_dataset)}")
        print("Testing...")
        model.load_state_dict(torch.load(model_weights_path))
        model = model.to(device)

        with torch.no_grad(), \
                tqdm(total=len(test_dataset)) as progress, \
                open(data_dir() / "submission.csv", "w") as csvfile:
            # Write the CSV header
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(["Id", "Category"])

            # Evaluate the model on our test split
            for data in test_loader:
                # Calculate outputs by running images through the network
                inputs, indicies = data
                inputs = inputs.to(device)
                outputs = model(inputs)

                # The class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs.data, 1)

                # Get the predicted name
                preds = predicted.cpu().tolist()
                indicies = indicies.tolist()
                for i, idx in enumerate(indicies):
                    prediction = dataset.get_label_desc(preds[i])
                    csvwriter.writerow([idx, prediction])
                    # print(f"Idx: {idx}, Category: {preds[i]}, Pred: {prediction}")

                progress.update(inputs.size(0))

    ############################## PLOT TRAINING LOSS ##############################
    if args.train:
        plt.figure(figsize=(10, 5))
        plt.title("Loss During Training")
        plt.plot(training_loss, label="loss")
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(str(output_dir() / model_weights_path.stem) + "_loss.png")
        plt.show()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        # do nothing here
        pass
