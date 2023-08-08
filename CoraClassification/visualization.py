import pandas as pd
import matplotlib.pyplot as plt
import os
from hydra import initialize, compose


def load_config(config_name, config_path) -> None:
    initialize(config_path=config_path)
    cfg = compose(config_name=config_name)

    return cfg


def get_csv_file(path: str):
    files = os.listdir(path)
    csv_files = []
    for file in files:
        if file.endswith(".csv"):
            csv_files.append(os.path.join(path, file))
    return csv_files


def csv_to_df(csv_files: list):
    return {os.path.basename(file): pd.read_csv(file) for file in csv_files}


def plot(
    dfs: list,
    features: list,
    is_save=False,
    fig_name="comparison",
    fig_path="visualization",
):
    fig = plt.figure(figsize=(20, 10))
    n_rows = 1
    n_cols = len(features)

    for i, feature in enumerate(features):
        fig.add_subplot(n_rows, n_cols, i + 1)
        legends = []
        for name, df in dfs.items():
            plt.plot(df["epoch"], df[feature])
            legends.append(name.replace(".csv", ""))
        plt.legend(legends)
        plt.xlabel("Epochs")
        plt.ylabel(feature.capitalize())
        plt.title(f"Visualize {feature} in training phase")
    if is_save:
        fig_path = os.path.join(fig_path, fig_name)
        plt.savefig(fig_path, dpi=300)
    plt.show()


if __name__ == "__main__":
    config = load_config("model.yaml", "./config")
    csv_files = get_csv_file("./train_result")
    dfs = csv_to_df(csv_files)
    plot(
        dfs,
        config.visualization.features,
        is_save=config.visualization.is_save,
        fig_name=config.visualization.fig_name,
        fig_path=config.visualization.fig_path,
    )
