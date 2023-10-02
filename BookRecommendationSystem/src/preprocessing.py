import pandas as pd
import re
from w3lib.html import replace_entities
import time
import math
import json

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


def filter_dict_by_id(dictionary: dict, ignore_id: list):
    new_dictionary = {}
    for key, value in dictionary.items():
        if key not in ignore_id:
            new_dictionary[key] = value

    return new_dictionary


def filter_data(
    df: pd.DataFrame,
    lower_bound: int,
    upper_bound: float = math.inf,
):
    users_id = []
    for user_id, group in df.groupby("user_id"):
        if upper_bound >= len(group) >= lower_bound:
            users_id.append(user_id)
    filtered_data = df[df["user_id"].isin(users_id)].reset_index()
    return filtered_data


def process_data(
    df: pd.DataFrame,
    is_save: bool = False,
    path: str = "BookReviewDataset",
    sample_rate: float = 1,
    minimum_interactions: int = 1,
):
    sample_size = int(sample_rate * len(df))
    df = df.sample(sample_size)
    df = filter_data(df, lower_bound=minimum_interactions)
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
        type_ = convert_size_to_tag(sample_rate)
        dst_path = f"{path}_{type_}.csv"
        preprocessed_data.to_csv(dst_path, index=False)

    books_id_convert = [str(key) for key in book_le.classes_]
    users_id_convert = [str(key) for key in user_le.classes_]
    le_user_mapping = dict(
        zip(
            users_id_convert,
            user_le.transform(user_le.classes_),
        )
    )

    le_book_mapping = dict(
        zip(
            books_id_convert,
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
        type_ = "XS"
    elif 0.15 <= sample_rates < 0.3:
        type_ = "S"
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
    sample_rate: float = 1,
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
        sample_rate=sample_rate,
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


def count_number_of_link(df: pd.DataFrame):
    count = {}
    for user_id, group in df.groupby("user_id"):
        count[user_id] = len(group)

    return count


def get_min_max_interaction(count: dict):
    count_sorted = sorted(count.items(), key=lambda x: x[1])
    return count_sorted[0][1], count_sorted[-1][1]


def describe(df: pd.DataFrame):
    number_of_users = df["user_id"].nunique()
    number_of_books = df["isbn"].nunique()
    number_of_links = len(df)
    count = count_number_of_link(df)
    min_interaction, max_interaction = get_min_max_interaction(count)

    return f"""The dataset's description
        + Number of users: {number_of_users:,}
        + Number of books: {number_of_books:,}
        + Number of links: {number_of_links:,}
        + Min interactions: {min_interaction:,}
        + Max interactions: {max_interaction:,}
    """


def save_description(content: str, path: str):
    with open(path, "w") as f:
        f.write(content.strip())


def get_data_by_ratio(
    raw_path: str,
    ratio: list[float],
    is_save: bool,
    save_path: str = "",
    minimum_interactions: int = 1,
):
    df = load_dataset(raw_path)
    df = preprocessing_data(df)
    datasets = []

    for sample_rate in ratio:
        print("Processing data...")
        t = time.process_time()
        (
            preprocessed_data,
            books_image,
            le_user_mapping,
            le_book_mapping,
        ) = process_data(
            df=df,
            is_save=is_save,
            path=save_path,
            sample_rate=sample_rate,
            minimum_interactions=minimum_interactions,
        )

        elapsed = time.process_time() - t
        type_ = convert_size_to_tag(sample_rate)
        print(f">>> The dataset size '{type_}' done! Time {elapsed}")
        save_description(
            describe(preprocessed_data),
            f"{save_path}_{type_}_description.txt",
        )

        datasets.append(
            (
                preprocessed_data,
                books_image,
                le_user_mapping,
                le_book_mapping,
            )
        )
    return datasets


if __name__ == "__main__":
    sample_rates = [0.02, 0.15, 0.3, 0.75]
    datasets = get_data_by_ratio(
        "../data/raw/Book-Crossing User review ratings.csv",
        [0.02],
        True,
        "../data/processed/BookReviewDataset_1",
        20,
    )
