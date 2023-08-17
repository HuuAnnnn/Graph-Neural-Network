import torch
from torch import nn
import torch.nn.functional as F


class GAT(nn.Module):
    def __init__(self, input_size, output_size, n_heads=4) -> None:
        super(GAT, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.n_heads = n_heads
        self.hidden = self.output_size // self.n_heads

        w = torch.empty((input_size, output_size))
        a = torch.empty((self.hidden * 2, 1))
        self.W = nn.Parameter(nn.init.kaiming_normal_(w))
        self.A = nn.Parameter(nn.init.kaiming_normal_(a))
        self.softmax = nn.Softmax(dim=1)

    def _edge_list_to_adjacency_matrix(self, edges, number_of_nodes):
        src, dst = edges[0], edges[0]
        adj = torch.zeros(
            (number_of_nodes, number_of_nodes),
            dtype=torch.float,
        )
        adj[src, dst] = 1

        return adj

    def forward(self, X, edges):
        number_of_nodes = X.shape[0]
        adj = self._edge_list_to_adjacency_matrix(
            edges=edges,
            number_of_nodes=number_of_nodes,
        ).to(X.device)
        adj = adj.view(*adj.size(), 1)
        self.to(X.device)
        # embedding
        h = X @ self.W
        h = h.view(number_of_nodes, self.n_heads, self.hidden)
        attn = self.cal_attention_score(embedding=h, adjacency_matrix=adj)
        return attn.reshape(number_of_nodes, self.n_heads * self.hidden)

    def cal_attention_score(self, embedding, adjacency_matrix):
        N = embedding.shape[0]
        embed_repeat = embedding.repeat(N, 1, 1)
        embed_repeat_interleave = embedding.repeat_interleave(N, dim=0)
        embed_concat = torch.concat(
            [embed_repeat_interleave, embed_repeat],
            dim=-1,
        )
        embed_concat = embed_concat.view(N, N, self.n_heads, 2 * self.hidden)
        e = embed_concat @ self.A
        leaky_relu = nn.LeakyReLU()
        e = leaky_relu(e)
        e = e.squeeze(-1)
        e = e.masked_fill_(adjacency_matrix == 0, float("-inf"))
        a = self.softmax(e)

        attn_res = torch.einsum("ijh,jhf->ihf", a, embedding)

        return attn_res

    def to(self, device=None):
        self.W = self.W.to(device)
        return super(GAT, self).to(device)

    def __repr__(self):
        return f"GAT({self.input_size}, {self.output_size})"
