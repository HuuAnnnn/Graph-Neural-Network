import torch
from torch import nn
from torch.nn import functional as F
from loguru import logger
from datetime import timedelta, datetime
from utils.early_stopping import EarlyStopping


class GNN(nn.Module):
    def __init__(
        self, input_size, hidden_size, output_size, base_layer=None
    ) -> None:
        """The module is a GNN model which is based on the GCN layer

        Parameters
        ----------
        input_size : The number of features of one node in the graph
        hidden_size : The latent dim for hidden layer
        output_size : the number of classes which is needed to classify
        """
        super(GNN, self).__init__()
        self.conv1 = base_layer(input_size, hidden_size)
        self.conv2 = base_layer(hidden_size, hidden_size)
        self.conv3 = base_layer(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, X, edges_index):
        out = X
        out = self.conv1(out, edges_index)
        out = out.relu()
        out = F.dropout(out, p=0.5, training=self.training)

        out = self.conv2(out, edges_index)
        out = out.relu()
        out = F.dropout(out, p=0.5, training=self.training)

        out = self.conv3(out, edges_index)
        out = out.relu()
        out = F.dropout(out, p=0.5, training=self.training)

        out = self.out(out)
        out = F.softmax(out, dim=1)
        return out

    def get_parameters(self):
        return sum(parameter.numel() for parameter in self.parameters())

    def compile(self, optimizer, loss):
        self.optimizer = optimizer
        self.criterion = loss

    def fit(self, epochs: int = 1, dataset=None, verbose: int = 1):
        assert self.optimizer != None
        assert self.criterion != None

        ers = EarlyStopping(patience=10, min_delta=0)
        history = {"epoch": [], "loss": [], "accuracy": [], "time": []}
        logger.critical("Start training!")
        for epoch in range(1, epochs + 1):
            self.train()
            epoch_start_time = datetime.now()
            self.optimizer.zero_grad()
            out = self.forward(dataset.x, dataset.edge_index)
            loss = self.criterion(
                out[dataset.train_mask],
                dataset.y[dataset.train_mask],
            )

            pred = torch.argmax(out[dataset.train_mask], dim=1)
            pred = pred[pred == dataset.y[dataset.train_mask]]
            accuracy = len(pred) / len(dataset.y[dataset.train_mask])

            loss.backward()
            self.optimizer.step()
            train_time_duration = datetime.now() - epoch_start_time
            history["epoch"].append(epoch)
            history["loss"].append(loss.cpu().detach().item())
            history["accuracy"].append(accuracy)
            history["time"].append(train_time_duration.total_seconds())

            if (epoch - 1) % verbose == 0:
                logger.info(
                    f"Epoch {epoch} \t| Time: {train_time_duration} \t| Loss: {loss.item()} \t| Accuracy: {accuracy}"
                )

            is_stop = ers(loss)

            if is_stop:
                logger.debug(f"Training stop at {epoch}")
                break
        logger.critical("Stop training!")

        return history

    def __str__(self):
        return (
            super(GNN, self).__str__()
            + f"\nNumber of parameters: {self.get_parameters():,}"
        )
