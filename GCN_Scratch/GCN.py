import torch
from torch import nn, sparse


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
        self.D = None
        self.I = None
        self.A = None

    def _extract_D_from_edge_list(self, edges):
        src, target = torch.tensor(edges[0]), torch.tensor(edges[1])
        all_nodes = torch.cat((src, target))
        unique_nodes, node_counts = torch.unique(all_nodes, return_counts=True)
        degrees = dict(zip(unique_nodes.tolist(), node_counts.tolist()))
        num_node = len(unique_nodes)
        D = torch.zeros((num_node, num_node))
        for node, degree in degrees.items():
            D[node][node] = degree
        return D

    def _edges_list_to_adj_matrix(self, edges_list, number_of_nodes):
        source_nodes, target_nodes = edges_list[0], edges_list[1]
        A = torch.zeros((number_of_nodes, number_of_nodes), dtype=torch.float)
        A[source_nodes, target_nodes] = 1
        return A

    def forward(self, X, edge_list):
        self.D = self._extract_D_from_edge_list(edges=edge_list).to(X.device)
        edge_list = edge_list.long().to(X.device)
        self.A = self._edges_list_to_adj_matrix(
            edges_list=edge_list,
            number_of_nodes=X.shape[0],
        )
        self.I = torch.eye(X.shape[0])
        self.to(X.device)
        A_hat = self.A + self.I
        z = torch.inverse(self.D) @ A_hat @ X @ self.W
        relu = nn.ReLU()
        return relu(z)

    def to(self, device=None):
        self.D = self.D.to(device)
        self.I = self.I.to(device)
        self.A = self.A.to(device)
        return super(GCN, self).to(device)

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
