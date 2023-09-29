import pandas as pd
import re
from w3lib.html import replace_entities
import time

from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer


def load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def preprocessing_data(df: pd.DataFrame):
    unnecessary_col = ["Unnamed: 0", "location"]
    df["Category"] = df["Category"].apply(
        lambda x: re.sub(r"[\"\[w+\]\']", "", x)
    )

    df["Summary"] = df["Summary"].apply(lambda x: replace_entities(x))
    df["Summary"] = df["Summary"].apply(
        lambda x: re.sub(r"[^a-zA-Z0-9\\\s]", "", x)
    )
    df = df.drop(unnecessary_col, axis=1)
    df = df.drop_duplicates(keep="first")

    # replace wrong data row
    df.drop(df[df["Language"] == "9"].index, inplace=True)
    df.drop(df[df["Category"] == "9"].index, inplace=True)
    df.drop(df[df["Summary"] == "9"].index, inplace=True)

    df = fill_missing_column(df)

    # replace wrong book's description found
    df = df.replace(
        "Alice (Fictitious character",
        "Alice (Fictitious character : Carroll)",
    )

    df = df.sort_values("user_id")

    return df


def process_data(
    df: pd.DataFrame,
    is_save: bool = False,
    path: str = "BookReviewDataset",
    sample_rates: float = 1,
):
    sample_size = int(sample_rates * len(df))
    df = df.sample(sample_size)
    encode_cols = [
        "book_title",
        "book_author",
        "publisher",
        "Language",
        "Category",
        "isbn",
        "city",
        "state",
        "country",
    ]

    for col in encode_cols:
        df.loc[:, col] = LabelEncoder().fit_transform(df[col])

    user_features_col = [
        "user_id",
        "age",
        "rating",
        "city",
        "state",
        "country",
    ]
    books_features_col = [
        "isbn",
        "book_title",
        "book_author",
        "year_of_publication",
        "publisher",
        "Summary",
        "Language",
        "Category",
    ]

    users_df = df[user_features_col].reset_index()
    books_df = df[books_features_col].reset_index()

    user_le = LabelEncoder()
    book_le = LabelEncoder()
    users_df["user_id"] = user_le.fit_transform(users_df["user_id"])
    books_df["isbn"] = book_le.fit_transform(books_df["isbn"])
    books_image = get_books_dict(books_df, df)

    # embedding description
    summary_df = embed_description_data(books_df, max_length=34)
    books_df = pd.concat([books_df, summary_df], axis=1)
    books_df.drop(["Summary"], inplace=True, axis=1)

    # combine data
    preprocessed_data = pd.concat([users_df, books_df], axis=1)
    preprocessed_data.drop(["index"], axis=1, inplace=True)

    if is_save:
        type_ = convert_size_to_tag(sample_rates)
        dst_path = f"{path}_{type_}.csv"
        preprocessed_data.to_csv(dst_path, index=False)
    le_user_mapping = dict(
        zip(
            user_le.classes_,
            user_le.transform(user_le.classes_),
        )
    )

    le_book_mapping = dict(
        zip(
            book_le.classes_,
            book_le.transform(book_le.classes_),
        )
    )

    return (
        preprocessed_data,
        books_image,
        le_user_mapping,
        le_book_mapping,
    )


def convert_size_to_tag(sample_rates: float):
    type_ = ""
    if 0 <= sample_rates < 0.15:
        type_ = "S"
    elif 0.15 <= sample_rates < 0.3:
        type_ = "XS"
    elif 0.3 <= sample_rates < 0.75:
        type_ = "M"
    else:
        type_ = "L"

    return type_


def embed_description_data(
    df: pd.DataFrame,
    max_length: int = 34,
    pre_train_path: str = "bert-base-uncased",
):
    summary_sentences = df["Summary"].to_list()
    tokenizer = BertTokenizer.from_pretrained(pre_train_path)
    encoding = tokenizer(
        summary_sentences,
        add_special_tokens=False,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_attention_mask=False,
        return_tensors="pt",
    )

    embedding_matrix = encoding["input_ids"]
    instance_view = embedding_matrix.transpose(0, 1)
    embeddings = {f"Summary_{i}": [] for i in range(1, max_length + 1)}

    for i, key in enumerate(embeddings.keys()):
        embeddings[key] = instance_view[i].tolist()

    summary_df = pd.DataFrame(embeddings)

    return summary_df


def pipeline(
    df: pd.DataFrame,
    is_save: bool = True,
    path: str = "./BookReviewDataset",
    sample_rates: float = 1,
):
    df = preprocessing_data(df)
    (
        preprocessed_data,
        books_image,
        le_user_mapping,
        le_book_mapping,
    ) = process_data(
        df=df,
        is_save=is_save,
        path=path,
        sample_rates=sample_rates,
    )

    return (
        preprocessed_data,
        books_image,
        le_user_mapping,
        le_book_mapping,
    )


def get_books_dict(books: pd.DataFrame, df: pd.DataFrame):
    books_images = {}
    for record in books.to_numpy():
        isbn = record[0]
        image_link = df.iloc[:, 8]
        if isbn not in books_images.keys():
            books_images[isbn] = image_link

    return books_images


def fill_missing_column(df: pd.DataFrame):
    for col in df.columns:
        if df[col].dtype != float:
            df[col] = df[col].fillna(df[col].mode()[0])

    return df


if __name__ == "__main__":
    sample_rates = [0.02, 0.15, 0.3, 0.75]

    for sample_rate in sample_rates:
        print("Processing data...")
        t = time.process_time()

        df = load_dataset("../data/raw/Book-Crossing User review ratings.csv")
        (
            preprocessed_data,
            books_image,
            le_user_mapping,
            le_book_mapping,
        ) = pipeline(
            df,
            path="../data/processed/BookReviewDataset",
            sample_rates=sample_rate,
        )
        elapsed = time.process_time() - t
        print(
            f">>> The dataset size '{convert_size_to_tag(sample_rate)}' done! Time {elapsed}"
        )
