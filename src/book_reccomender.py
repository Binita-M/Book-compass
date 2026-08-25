import pathlib
from typing import Tuple, Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def book_compass(csv_path, k=10, top_n_users_to_show=0):
    #Load the ratings dataset and explore its structure
    df = pd.read_csv('book_ratings.csv')
    parameters = {"user_id", "book_title", "rating"}
    missing = parameters - set(df.columns)
    print(f"parameters: {parameters}")
    print(f"df.columns as set: {set(df.columns)}")
    print(f"missing: {missing}")
    if missing:
        raise ValueError(f"CSV is missing columns: {missing}")

    rating = len(df)
    users = df['user_id'].nunique()
    books = df['book_title'].nunique()
    rating_range = (df['rating'].min(), df['rating'].max())

    print(f"Total ratings: {rating}")
    print(f"Unique users: {users}")
    print(f"Unique books: {books}")
    print(f"Rating range: {rating_range}")

    #Initializing a matrix of nan, filling in the observed ratings from the dataframe and counting nan values and computing sparsity
    user_ids = sorted(df['user_id'].unique())
    book_ids = sorted(df['book_title'].unique())

    user_to_idx = {user: i for i, user in enumerate(user_ids)}
    book_to_idx = {book: i for i, book in enumerate(book_ids)}

    rating_matrix = np.full((len(user_ids), len(book_ids)), np.nan)
    user_indices = df['user_id'].map(user_to_idx).values
    book_indices = df['book_title'].map(book_to_idx).values

    rating_matrix[user_indices,book_indices] = df['rating'].values
    missing = int(np.isnan(rating_matrix).sum())
    sparsity = round(float(missing/rating_matrix.size), 3)

    print(f"Rating matrix shape: {rating_matrix.shape}")
    print(f"Missing_entries: {missing}")
    print(f"Sparsity: {sparsity:.1%}")

    #Visualize the sparsity pattern
    fig, ax = plt.subplots(figsize=(12, 8))
    mask = ~np.isnan(rating_matrix)
    ax.imshow(mask, aspect="auto", cmap="Blues", interpolation="nearest")
    ax.set_xlabel("Book Title")
    ax.set_ylabel("User ID")
    ax.set_title(f"Rating Matrix sparsity pattern (white = missing, blue = observed)\nSparsity: {sparsity:.1%}")
    plt.tight_layout()
    plt.show()

    #Centering the rating matrix by book means
    book_means = np.nanmean(rating_matrix, axis=0)
    centered_matrix = rating_matrix - book_means
    centered_mean = float(np.nanmean(centered_matrix))

    print(f"Book means: {book_means[:5].round(2)}")
    print(f"Centered mean : {centered_mean}")

    #Performing mean imputation on the centered matrix
    mean_imputed = np.nan_to_num(centered_matrix, nan=0.0)
    has_nan = bool(np.isnan(mean_imputed).any()) #sanity check
    imputed_range = (float(mean_imputed.min()),float(mean_imputed.max()))
    print(f"Imputed matrix shape: {mean_imputed.shape}")
    print(f"Contains NaN?: {has_nan}")
    print(f"Range: {imputed_range}")
    return {
        "mean_imputed": mean_imputed,
        "rating_matrix": rating_matrix,
        "centered_matrix": centered_matrix,
        "book_means": book_means,
        "user_to_idx": user_to_idx,
        "book_to_idx": book_to_idx,
        "user_ids": user_ids,
        "book_ids": book_ids,
    }

book_compass('book_ratings.csv', k=10, top_n_users_to_show=0)

