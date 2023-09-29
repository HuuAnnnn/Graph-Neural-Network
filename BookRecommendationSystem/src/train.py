import torch
import pandas as pd
import pandas as pd

import torch.nn.functional as F

from torch_geometric import nn
from torch_geometric import transforms as T
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.data import HeteroData
from torch_geometric.utils import negative_sampling

from sklearn.preprocessing import MinMaxScaler


class BookReview:
    """Book Review dataset
    Attributes:
        + user_id: user's id (0 -> 67698)
        + age: user'age
        + city: user's city location
        + state: user's state location
        + country: user's country location
        + isbn: book identify (67698 -> 138700)
        + book_title: book's title
        + book_author: book's author
        + year_of_publication: books's publication year
        + rating: the score user rate for the book
        + Summary_{1,24}: the description of book embedded by BERT
    """

    def __init__(self, path: str) -> None:
        self._path = path

    def _load_dataset(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        summary_attr = [f"Summary_{i}" for i in range(1, 35)]
        user_attributes = ["user_id", "age", "city", "state", "country"]
        book_attributes = [
            "isbn",
            "book_title",
            "book_author",
            "year_of_publication",
        ] + summary_attr

        df = pd.read_csv(self._path)
        users_df = df[user_attributes].reset_index(drop=True)
        books_df = df[book_attributes].reset_index(drop=True)
        rating_df = df[["user_id", "isbn", "rating"]].reset_index(drop=True)

        return users_df, books_df, rating_df

    def __call__(self) -> HeteroData:
        data = HeteroData()
        users_df, books_df, rating_df = self._load_dataset()
        y = torch.from_numpy(rating_df["rating"].to_numpy())
        edge_index = torch.from_numpy(
            rating_df[["user_id", "isbn"]].values.transpose()
        )

        data.name = "Book rating"
        data["users"].node_id = torch.from_numpy(
            users_df["user_id"].values
        ).to(dtype=torch.int64)
        data["books"].node_id = torch.from_numpy(books_df["isbn"].values).to(
            dtype=torch.int64
        )
        data.number_of_users = len(users_df["user_id"].unique())
        data.number_of_books = len(books_df["isbn"].unique())
        books_df.drop(["isbn"], inplace=True, axis=1)
        users_df.drop(["user_id"], inplace=True, axis=1)
        data.number_of_nodes = data.number_of_users + data.number_of_books
        data.number_of_user_node_features = len(users_df.columns)
        data.number_of_book_node_features = len(books_df.columns)

        feat_users_scaler = MinMaxScaler().fit_transform(users_df.to_numpy())
        feat_books_scaler = MinMaxScaler().fit_transform(books_df.to_numpy())
        data["users"].x = torch.from_numpy(feat_users_scaler)
        data["books"].x = torch.from_numpy(feat_books_scaler)
        data["users", "rating", "books"].edge_index = edge_index
        data["users", "books"].edge_label = y
        return data


class HyperParameters:
    BATCH_SIZE = 512
    EPOCHS = 100
    LEARNING_RATE = 0.005


if __name__ == "__main__":
    book_review_dataset = BookReview(
        path="../data/processed/BookReviewProcessedData_S.csv"
    )
    dataset = book_review_dataset()
    dataset = T.ToUndirected()(dataset)

    transform = T.RandomLinkSplit(
        num_val=0.1,
        num_test=0.1,
        disjoint_train_ratio=0.3,
        neg_sampling_ratio=1.0,
        add_negative_train_samples=True,
        edge_types=("users", "rating", "books"),
        rev_edge_types=("books", "rev_rating", "users"),
        split_labels=True,
    )

    train, val, test = transform(dataset)
    edge_label_index = train["users", "rating", "books"].pos_edge_label_index
    edge_label = train["users", "rating", "books"].pos_edge_label

    train_loader = LinkNeighborLoader(
        data=train,
        num_neighbors=[20, 10],
        neg_sampling_ratio=2.0,
        edge_label_index=(("users", "rating", "books"), edge_label_index),
        edge_label=edge_label,
        batch_size=HyperParameters.BATCH_SIZE,
        shuffle=True,
    )
    
    print(train_loader)
