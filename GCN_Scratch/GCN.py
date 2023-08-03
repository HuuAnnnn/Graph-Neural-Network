import torch
from torch import nn


class GCN(nn.Module):
    """Implementation of Graph convolution network based on the paper Semi-Supervised Classification with Graph Convolutional Networks"""

    def __init__(
        self,
        input_size: int,
        output_size: int,
    ) -> None:
        super(GCN, self).__init__()
        w = torch.empty(input_size, output_size)
        self.W = nn.Parameter(nn.init.kaiming_normal_(w))

    def _extract_D_I_from_edge_list(self, edges, number_of_nodes):
        D = torch.zeros((number_of_nodes, number_of_nodes))
        edges_list = list(zip(*edges))
        for edge in edges_list:
            src, dst = edge
            D[src][src] += 1
            D[dst][dst] += 1

        return D

    def _edge_list_to_adjacency_matrix(self, edges, number_of_nodes):
        A = torch.zeros((number_of_nodes, number_of_nodes))
        edges_list = list(zip(*edges))
        for src, dst in edges_list:
            A[src][dst] = 1
            A[dst][src] = 1

        return A

    def forward(self, X, edge_list):
        A = self._edge_list_to_adjacency_matrix(
            edges=edge_list,
            number_of_nodes=X.shape[0],
        )

        D = self._extract_D_I_from_edge_list(
            edges=edge_list,
            number_of_nodes=X.shape[0],
        )

        I = torch.eye(X.shape[0])
        A_hat = A + I
        z = torch.inverse(D) @ A_hat @ X @ self.W
        relu = nn.ReLU()
        return relu(z)

    def __str__(self):
        input_size, output_size = self.W.shape
        return f"{GCN.__name__}({input_size}, {output_size})"

    def __repr__(self):
        input_size, output_size = self.W.shape
        return f"{GCN.__name__}({input_size}, {output_size})"


if __name__ == "__main__":
    X = torch.FloatTensor(
        [
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0],
            [0, 1, 1],
        ]
    )
    gcn = GCN(3, 1)
    print(gcn)
    print(gcn(X, [[0, 1, 1, 2], [1, 2, 3, 3]]))
