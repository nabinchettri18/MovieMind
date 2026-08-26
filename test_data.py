from src.data_loader import load_movies, load_ratings, load_users


movies = load_movies()
ratings = load_ratings()
users = load_users()

print("Movies:", movies.shape)
print("Ratings:", ratings.shape)
print("Users:", users.shape)

print("\nFirst 5 Movies:")
print(movies.head())

print("\nFirst 5 Ratings:")
print(ratings.head())

print("\nFirst 5 Users:")
print(users.head())