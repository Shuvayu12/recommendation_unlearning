
import os
import zipfile
import pandas as pd
import numpy as np
import requests

def download_movielens(dest_folder="movielens_data", url="https://files.grouplens.org/datasets/movielens/ml-1m.zip"):
    os.makedirs(dest_folder, exist_ok=True)
    zip_path = os.path.join(dest_folder, "ml-1m.zip")
    if not os.path.exists(zip_path):
        print("Downloading MovieLens 1M dataset...")
        r = requests.get(url)
        with open(zip_path, "wb") as f:
            f.write(r.content)
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(dest_folder)
    print("Downloaded and extracted MovieLens 1M.")

def preprocess_movielens(dest_folder="movielens_data"):
    ratings_path = os.path.join(dest_folder, "ml-1m", "ratings.dat")
    users_path = os.path.join(dest_folder, "ml-1m", "users.dat")
    movies_path = os.path.join(dest_folder, "ml-1m", "movies.dat")
    ratings = pd.read_csv(ratings_path, sep='::', engine='python', names=["UserID", "MovieID", "Rating", "Timestamp"])
    user2idx = {u: i for i, u in enumerate(ratings["UserID"].unique())}
    item2idx = {m: i for i, m in enumerate(ratings["MovieID"].unique())}
    ratings["UserIdx"] = ratings["UserID"].map(user2idx)
    ratings["ItemIdx"] = ratings["MovieID"].map(item2idx)
    ratings = ratings.sort_values(["UserIdx", "Timestamp"])
    train, test = [], []
    for uid, group in ratings.groupby("UserIdx"):
        if len(group) < 2:
            train.append(group)
        else:
            train.append(group.iloc[:-1])
            test.append(group.iloc[-1:])
    train = pd.concat(train)
    test = pd.concat(test)
    train.to_csv(os.path.join(dest_folder, "train.csv"), index=False)
    test.to_csv(os.path.join(dest_folder, "test.csv"), index=False)
    print("Preprocessing complete. Train and test files saved.")

if __name__ == "__main__":
    download_movielens()
    preprocess_movielens()
