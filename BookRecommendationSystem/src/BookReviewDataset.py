import pandas as pd
from torch_geometric.data import HeteroData


class BookReviewDataset:
    def __init__(self) -> None:
        pass

    def _load_dataset(
        self,
        path: str,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        summary_attr = [f"Summary_{i}" for i in range(1, 35)]
        user_attributes = ["user_id", "age", "city", "state", "country"]
        book_attributes = [
            "isbn",
            "book_title",
            "book_author",
            "year_of_publication",
        ] + summary_attr
        df = pd.read_csv(path)

        users_df = df[user_attributes].reset_index(drop=True)
        books_df = df[book_attributes].reset_index(drop=True)
        rating_df = df[["user_id", "isbn", "rating"]].reset_index(drop=True)

        return users_df, books_df, rating_df

    def __call__(self, path: str) -> HeteroData:
        users_df, books_df, rating_df = self._load_dataset(path)
        y = rating_df["rating"].to_numpy()
        data = HeteroData()
        edge_index = rating_df[["user_id", "isbn"]].values.transpose()
        data["users"].x = users_df.to_numpy()
        data["books"].x = books_df.to_numpy()
        data["users", "rating", "books"].edge_index = edge_index
        data["users", "books"].y = y
        
        return data
