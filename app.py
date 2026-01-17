import streamlit as st
import pandas as pd
import joblib
import numpy as np

# -----------------------
# Load everything once
# -----------------------
@st.cache_resource
def load_assets():
    ratings = pd.read_csv("data/ratings.csv")
    movies = pd.read_csv("data/movies.csv")
    algo = joblib.load("models/svd_model.pkl")
    cosine_sim = joblib.load("models/cosine_sim.pkl")
    return ratings, movies, algo, cosine_sim

ratings, movies, algo, cosine_sim = load_assets()

st.title("🎬 Movie Recommendation System")

# -----------------------
# Inputs
# -----------------------
user_id = st.number_input("Enter User ID", min_value=1, step=1)
movie_title = st.selectbox("Pick a movie you like", movies['title'])

rec_type = st.radio(
    "Recommendation Type",
    ("Content-Based", "Collaborative Filtering", "Hybrid")
)

# -----------------------
# CONTENT-BASED
# -----------------------
def content_recs(title, topn=10):
    idx = movies[movies['title'] == title].index[0]
    scores = list(enumerate(cosine_sim[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:topn+1]
    indices = [i[0] for i in scores]
    return movies.iloc[indices][['title', 'genres']]

# -----------------------
# COLLABORATIVE
# -----------------------
def collab_recs(user_id, topn=10):
    seen = ratings[ratings.userId == user_id]['movieId'].unique()
    unseen = movies[~movies.movieId.isin(seen)]

    preds = []
    for mid in unseen['movieId']:
        preds.append((mid, algo.predict(user_id, mid).est))

    preds = sorted(preds, key=lambda x: x[1], reverse=True)[:topn]
    return movies[movies.movieId.isin([p[0] for p in preds])][['title', 'genres']]

# -----------------------
# HYBRID
# -----------------------
def hybrid_recs(user_id, title, alpha=0.6, topn=10):
    idx = movies[movies['title'] == title].index[0]
    scores = []

    for i, row in movies.iterrows():
        if i == idx:
            continue

        content_score = cosine_sim[idx][i]
        collab_score = algo.predict(user_id, row['movieId']).est

        final_score = alpha * collab_score + (1 - alpha) * content_score
        scores.append((i, final_score))

    scores = sorted(scores, key=lambda x: x[1], reverse=True)[:topn]
    return movies.iloc[[s[0] for s in scores]][['title', 'genres']]

# -----------------------
# UI Action
# -----------------------
if st.button("Recommend 🎯"):
    if rec_type == "Content-Based":
        st.write(content_recs(movie_title))
    elif rec_type == "Collaborative Filtering":
        st.write(collab_recs(user_id))
    else:
        st.write(hybrid_recs(user_id, movie_title))
