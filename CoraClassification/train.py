import torch
from torch import nn, optim
from torch import nn
from torch.nn import functional as F

from torch_geometric import datasets
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.nn import GCNConv

import pandas as pd
import os
import sys
from hydra import compose, initialize
from loguru import logger

from models import GNN

sys.path.append(os.path.abspath(os.path.join("..", "GCN")))
from GCN import GCN


def load_config(config_name, config_path) -> None:
    initialize(config_path=config_path)
    cfg = compose(config_name=config_name)

    return cfg


def load_model(model_name):
    models = {"torch": GCNConv, "scratch": GCN}
    return models[model_name]


def load_dataset(config):
    dataset = datasets.Planetoid(
        root=config.dataset.save_path,
        name=config.dataset.name,
        transform=NormalizeFeatures(),
    )

    return dataset


def describe(dataset):
    print("Dataset describe:")
    for name, value in dataset.__dict__.items():
        if isinstance(value, str):
            print(f"\t{name.capitalize()}: {value}")

    for name, value in dataset[0].__dict__["_store"].items():
        if torch.is_tensor(value):
            print(f"\t{name.capitalize()}: {list(value.shape)}")


def build_model(config, input_size, hidden_size, output_size):
    base_layer = load_model(config.model.name)
    model = GNN.GNN(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        base_layer=base_layer,
    )
    return model


def save_history(path, name, history):
    if not os.path.exists(path):
        os.makedirs(path)

    history_df = pd.DataFrame(history)
    history_df.to_csv(os.path.join(path, f"{name}.csv"))


if __name__ == "__main__":
    config = load_config(config_path="./config", config_name="model.yaml")
    assert config != None, logger.debug("Failed to load configuration")
    logger.info("The configuration is loaded")

    dataset_raw = load_dataset(config)
    dataset = dataset_raw[0]
    describe(dataset_raw)

    model = build_model(
        config,
        dataset_raw.num_node_features,
        config.model.latent_dim,
        dataset_raw.num_classes,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = dataset.to(device=device)
    model = model.to(device=device)
    logger.debug(f"Use '{device}' for training")

    optimizer = optim.Adam(model.parameters())
    loss = nn.CrossEntropyLoss()

    model.compile(optimizer=optimizer, loss=loss)
    history = model.fit(config.training.epochs, dataset=dataset)
    save_history(
        path=config.training.save_path,
        name=f"{config.dataset.name}_{config.model.name}",
        history=history,
    )

    logger.critical("Saved history")
