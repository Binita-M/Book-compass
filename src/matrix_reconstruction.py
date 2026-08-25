import pathlib
from typing import Tuple, Dict, List, Optional
from book_reccomender import book_compass 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.linalg import svd

RANDOM_STATE = 42


def svd_reconstruct_matrix(matrix, rank):
    U, S, Vt = np.linalg.svd(matrix, full_matrices=False)
    U_M = U[:, :rank]
    S_M = S[:rank]
    Vt_M = Vt[:rank, :]
    reconstruction = U_M @ np.diag(S_M) @Vt_M
    return reconstruction

#Implementing the iterative matrix completion algorithm
def complete_matrix(original_matrix, rank, max_iter=20, tol=1e-4):
    observed_mask = ~np.isnan(original_matrix)
    current_matrix = mean_imputed.copy()

    errors = []

    for iteration in range(max_iter):
        reconstructed = svd_reconstruct_matrix(current_matrix, rank)
        new_matrix = np.where(observed_mask, current_matrix, reconstructed)
        error = np.sqrt(np.mean((new_matrix - current_matrix) **2))
        errors.append(float(error))
        current_matrix = new_matrix

        if error < tol:
            break

    return current_matrix, errors

if __name__ == "__main__":
    results = book_compass('book_ratings.csv', k=10, top_n_users_to_show=0)
    mean_imputed = results["mean_imputed"]
    rating_matrix = results["rating_matrix"]
    centered_matrix = results["centered_matrix"]
    user_to_idx = results["user_to_idx"]
    book_to_idx = results["book_to_idx"]
    user_ids = results["user_ids"]
    book_ids = results["book_ids"]
    book_means = results["book_means"]
 
    reconstructed = svd_reconstruct_matrix(mean_imputed, 5)
    reconstruction_error = round(float(np.sqrt(np.mean((mean_imputed - reconstructed) ** 2))), 4)

    print(f"Reconstruction Matrix Shape: {reconstructed.shape}")
    print(f"Reconstruction RMSE: {reconstruction_error}")

    completed, errors = complete_matrix(rating_matrix, rank=5, max_iter=20)

    reconstructed_final = svd_reconstruct_matrix(completed, 5)
    observed_mask = ~np.isnan(rating_matrix)
    final_error = round(float(np.sqrt(np.mean((centered_matrix[observed_mask] - reconstructed_final[observed_mask]) **2))), 4)

    print(f"Completed matrix shape: {completed.shape}")
    print(f"Number of iterations: {len(errors)}")
    print(f"Final RMSE on observed: {final_error}")

    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(range(1, len(errors) + 1), errors, "o-", linewidth=2, markersize=6)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("RMSE on Observed Entries")
    ax.set_title("Convergence of Iterative Matrix Completion")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    #Selecting optimal rank using validation by masking
    np.random.seed(RANDOM_STATE)

    observed_mask_full = ~np.isnan(rating_matrix)
    observed_indices = np.argwhere(observed_mask_full)

    n_observed = len(observed_indices)
    n_validation = int(0.2 * n_observed)

    val_selection = np.random.choice(n_observed, size=n_validation, replace=False)
    val_indices = observed_indices[val_selection]

    validation_mask = np.zeros_like(observed_mask_full, dtype=bool)
    validation_mask[val_indices[:, 0], val_indices[:, 1]] = True

    train_mask = observed_mask_full & ~validation_mask

    train_matrix = rating_matrix.copy()
    train_matrix[validation_mask] = np.nan

    candidate_ranks = [1,2,3,5,10]
    validation_errors = {}

    for rank in candidate_ranks:
        train_means = np.nanmean(train_matrix, axis=0)
        train_centered = train_matrix - train_means
        train_imputed = np.nan_to_num(train_centered, nan=0.0)

        completed, _ = complete_matrix(train_matrix, rank=rank, max_iter=20)
        reconstructed_ratings = completed + train_means

        true_values = rating_matrix[validation_mask]
        predicted_values = reconstructed_ratings[validation_mask]

        rmse = np.sqrt(np.mean((true_values - predicted_values) **2))
        validation_errors[rank] = round(float(rmse), 4)
    
    best_rank = min(validation_errors, key=validation_errors.get)
    best_error = float(validation_errors[best_rank])

    print(f"Best rank: {best_rank}")
    print(f"Best error: {best_error}")
    print(validation_errors)

    #visualizing validation errors by rank
    fig, ax = plt.subplots(figsize=(8,5))
    ranks = sorted(validation_errors.keys())  
    errors = [validation_errors[r] for r in ranks]
    ax.plot(ranks, errors, "o-", linewidth=2, markersize=8)
    ax.axvline(best_rank, color="red", linestyle="--", alpha=0.7, label=f"Best: {best_rank}")
    ax.set_xlabel("Rank(M)") 
    ax.set_ylabel("Validation RMSE")     
    ax.set_title("Rank selection via validation")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

def predict_rating(user_id, book_title, completed_matrix, book_means):
    user_idx = user_to_idx[user_id]
    book_idx = book_to_idx[book_title]
    predicted = completed_matrix[user_idx, book_idx] + book_means[book_idx]
    predicted = np.clip(predicted, 1.0, 5.0)
    return round(float(predicted), 3)

final_completed = completed
user_0_id = user_ids[0]
user_0_unrated_indices = np.where(np.isnan(rating_matrix[user_to_idx[user_0_id]]))[0]
unrated_books = set(book_ids[m_idx] for m_idx in user_0_unrated_indices)
user_0_predictions = {book_ids[m_idx]: predict_rating(user_0_id, book_ids[m_idx], final_completed, book_means) for m_idx in user_0_unrated_indices}
sorted_predictions = sorted(user_0_predictions.items(), key=lambda x: x[1], reverse=True)
top_5_books = [book_id for book_id, rating in sorted_predictions[:5]]

print(f"User 0 has {len(user_0_predictions)} unrated books")
print(f"Top 5 movie recommendations for user 0: {top_5_books}")
print("Predicted ratings for top 5:")
for book_id in top_5_books:
    print(f"Book {book_id}: {user_0_predictions[book_id]:.2f}")
