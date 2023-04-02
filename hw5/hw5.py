import argparse

import torch
from torch.utils.data import random_split, DataLoader
import torch.nn.functional as Fun
import torchvision
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names
from torchsummary import summary
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from timm.models import InceptionV4, InceptionResnetV2
# SEE: https://github.com/huggingface/pytorch-image-models/blob/7501972cd61dde7428164041b0a6dd8fea60c4d4/timm/layers/adaptive_avgmax_pool.py#L124
from timm.models.layers import SelectAdaptivePool2d
from PIL import Image
from tqdm import tqdm

from utils import data_dir, count_trainable_parameters
from utils import FaceDataset, Trainer, FeatureModel


class ClassificationModel(torch.nn.Module):
    """Model that we will be training to recognize our faces"""

    def __init__(self, output_classes: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.model = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d(output_size=1),
            SelectAdaptivePool2d(flatten=True),
            torch.nn.Linear(in_features=1536,
                            out_features=output_classes, bias=False),
            torch.nn.LogSoftmax(dim=1)  # For NLL LOSS
        )

    def forward(self, x):
        return self.model(x)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", help="train the model",
                        action="store_true")
    parser.add_argument("--val", help="validate the model on testing split of data",
                        action="store_true")
    args = parser.parse_args()

    # Define device for pytorch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {str(device).upper()}")
    # Random seed
    seed = torch.Generator().manual_seed(43)

    ############################## PRE-TRAINED MODEL SETUP ##############################
    # Basic idea here is to take a pretrained classification model and remove the last
    # few layers (e.g. flatten, pooling, softmax). Then re-attach the last few layers
    # with "requires_grad" = True and retrain the last few layers using our training data.

    # List all available timm pretrained models
    # print(timm.list_models(pretrained=True))

    # SEE: https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/inception_v4.py
    model: InceptionResnetV2 = timm.create_model(
        "inception_resnet_v2", pretrained=True)
    # Set model to eval mode for inference
    model.eval()
    # Freeze gradient calculation (we will not be training this model)
    for param in model.parameters():
        param.requires_grad = False

    # # Create a new model that gets the output of the second to last layer
    # feature_model = FeatureModel(model, "global_pool.flatten")
    # # feature_model = FeatureModel(model, "features.21.cat_2")
    # feature_model.eval()

    # Setup data transformation for input
    config = resolve_data_config({}, model=model)
    # print(config)
    transform = create_transform(**config)
    # print(transform)

    ############################## LOAD DATASETS ##############################

    dataset_string = "train_small"
    dataset = FaceDataset(
        csv_file=data_dir() / f"{dataset_string}.csv",
        root_dir=data_dir() / f"{dataset_string}",
        transform=transform
    )

    # Split the training data into train + test datasets
    train_dataset, test_dataset = random_split(
        dataset, [0.8, 0.2], generator=seed)
    print("Dataset Split - Train: {}, Test: {}, Total: {}".format(
        len(train_dataset), len(test_dataset), len(dataset)
    ))

    batch_size = 4
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    ############################## SHOW DATASET SAMPLE ##############################

    # Show a few images
    # train_iter = iter(train_loader)
    # images, labels = next(train_iter)

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

    ############################## CREATE CLASSIFICATION MODEL ##############################

    # Create custom classification model
    classifier = ClassificationModel(output_classes=dataset.num_categories())

    # Define a loss function and an optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.01, momentum=0.6)
    # Decay LR by a factor of *gamma* every *step_size* epochs
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=7, gamma=0.1)

    ############################## TRAIN THE CLASSIFIER ##############################

    model_weights_path = data_dir() / f"{dataset_string}_weights.pth"
    if args.train:
        trainer = Trainer(dataset.num_categories(), 1)
        trainer.train(train_loader, model,
                      criterion, optimizer, scheduler)

        # Save the model weights
        torch.save(model.state_dict(), model_weights_path)

    ############################## VALIDATE THE CLASSIFIER ##############################

    if args.val:
        print("Validating...")
        # net = ClassificationModel(
        #     output_classes=dataset.num_categories())
        model.load_state_dict(torch.load(model_weights_path))

        # Evaluate the model on our test split
        total = 0
        correct = 0
        with tqdm(total=len(test_dataset)) as progress:
            for data in test_loader:
                # Calculate outputs by running images through the network
                inputs, labels = data
                outputs = model(inputs)
                # The class with the highest energy is what we choose as prediction
                torch.max(outputs, 1)

                batch_count = labels.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += batch_count
                correct += (predicted == labels).sum().item()
                progress.update(batch_count)
        print(
            f'Accuracy on the {total} test images: {100 * correct // total} %')


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        # do nothing here
        pass
