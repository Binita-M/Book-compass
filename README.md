BookCompass 📚

BookCompass is a Python-based book recommendation system built on collaborative filtering with matrix completion. It predicts how a user would rate books they haven't read yet, using an iterative SVD (Singular Value Decomposition) reconstruction approach.

How It Works

Ratings matrix — User ratings are organized into a user × book matrix, where missing (unrated) entries are NaN.

Mean-centering — Each book's average rating is subtracted out, so the matrix represents how much a user's rating deviates from a book's average.

Iterative SVD completion — The matrix is repeatedly reconstructed via truncated SVD, filling in missing entries with reconstructed values while keeping observed ratings fixed, until the reconstruction converges.

Rank selection via validation — A portion of observed ratings is masked out and used as a validation set to choose the best-performing rank (number of latent factors) by RMSE.

Prediction — For any user/book pair, the completed matrix value is added back to the book's mean rating (and clipped to the 1–5 range) to produce a final predicted rating.

Recommendations — For a given user, all unrated books are ranked by predicted rating, and the top N are returned as recommendations.
