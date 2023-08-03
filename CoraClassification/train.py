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

from models import GNN

sys.path.append(os.path.abspath(os.path.join("..", "GCN_scratch")))
from GCN import GCN


def load_config(config_name, config_path) -> None:
    initialize(config_path=config_path)
    cfg = compose(config_name=config_name)

    return cfg


def load_model(model_name):
    models = {"torch": GCNConv, "scratch": GCN}
    return models[model_name]


def load_dataset(config):
    cora = datasets.Planetoid(
        root=config.dataset.save_path,
        name=config.dataset.name,
        transform=NormalizeFeatures(),
    )

    return cora


def build_model(config, input_size, hidden_size, output_size):
    base_layer = load_model(config.model.name)
    model = GNN.GNN(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        base_layer=base_layer,
    )
    return model


if __name__ == "__main__":
    config = load_config(config_path="./config", config_name="model.yaml")
    assert config != None, Exception("Failed to load configuration")

    cora = load_dataset(config)
    dataset = cora[0]

    model = build_model(
        config,
        cora.num_node_features,
        config.model.latent_dim,
        cora.num_classes,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset.to(device=device)
    model.to(device=device)

    optimizer = optim.Adam(model.parameters())
    loss = nn.CrossEntropyLoss()

    model.compile(optimizer=optimizer, loss=loss)
    model.fit(config.training.epochs, dataset=dataset)
