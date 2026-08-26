import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

from src.data_loader import load_movies, load_ratings
from src.svd_recommender import SVDRecommender


# --------------------------------------------------
# Load ratings
# --------------------------------------------------

ratings = load_ratings()

print("Total ratings:", len(ratings))


# --------------------------------------------------
# Train / test split
# --------------------------------------------------

train_ratings, test_ratings = train_test_split(
    ratings,
    test_size=0.20,
    random_state=42
)

print("Training ratings:", len(train_ratings))
print("Testing ratings:", len(test_ratings))


# --------------------------------------------------
# Train SVD model
# --------------------------------------------------

print("\nTraining SVD model...")

model = SVDRecommender(
    train_ratings,
    load_movies(),
    factors=50
)

print("Model trained!")


# --------------------------------------------------
# Generate predictions
# --------------------------------------------------

actual = []
predicted = []

print("\nEvaluating model...")

for _, row in test_ratings.iterrows():

    prediction = model.predict(
        int(row["userId"]),
        int(row["movieId"])
    )

    if prediction is not None:

        actual.append(
            row["rating"]
        )

        predicted.append(
            prediction
        )


# --------------------------------------------------
# Calculate metrics
# --------------------------------------------------

rmse = np.sqrt(
    mean_squared_error(
        actual,
        predicted
    )
)

mae = mean_absolute_error(
    actual,
    predicted
)


# --------------------------------------------------
# Results
# --------------------------------------------------

print("\n==============================")
print(" MovieMind SVD Evaluation")
print("==============================")

print(
    f"Evaluated ratings: {len(actual)}"
)

print(
    f"RMSE: {rmse:.4f}"
)

print(
    f"MAE:  {mae:.4f}"
)

print("==============================")