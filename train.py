import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy

# -----------------------
# Load data
# -----------------------
ratings = pd.read_csv("data/ratings.csv")
movies = pd.read_csv("data/movies.csv")

print("Data loaded")

# -----------------------
# CONTENT-BASED MODEL
# -----------------------
movies['genre_text'] = movies['genres'].str.replace('|', ' ')

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['genre_text'])

cosine_sim = cosine_similarity(tfidf_matrix)

joblib.dump(tfidf, "models/tfidf.pkl")
joblib.dump(cosine_sim, "models/cosine_sim.pkl")

print("Content-based model saved")

# -----------------------
# COLLABORATIVE FILTERING
# -----------------------
reader = Reader(rating_scale=(ratings.rating.min(), ratings.rating.max()))
data = Dataset.load_from_df(
    ratings[['userId', 'movieId', 'rating']],
    reader
)

trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

algo = SVD(
    n_factors=50,
    n_epochs=20,
    reg_all=0.02,
    random_state=42
)

algo.fit(trainset)

preds = algo.test(testset)
rmse = accuracy.rmse(preds)
print("RMSE:", rmse)

joblib.dump(algo, "models/svd_model.pkl")
print("Collaborative model saved")
