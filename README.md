# 🎬 Movie Recommendation System

A full-fledged **Movie Recommendation System** built using Machine Learning techniques, featuring:

- **Content-Based Filtering**
- **Collaborative Filtering (Matrix Factorization – SVD)**
- **Hybrid Recommendation System**
- Interactive **Streamlit Web App**

This project demonstrates an end-to-end ML workflow — from data preprocessing and model training to deployment-ready inference.

---

## 🚀 Features

- 📌 Content-based recommendations using **TF-IDF + Cosine Similarity**
- 👥 Collaborative filtering using **SVD (Surprise library)**
- 🔀 Hybrid recommender combining content-based & collaborative scores
- 🖥️ Clean and interactive **Streamlit UI**
- 🧠 Proper separation of **training** and **inference**
- 📦 Production-friendly project structure

---

## 🛠️ Tech Stack

- **Python**
- **pandas, NumPy**
- **scikit-learn**
- **scikit-surprise**
- **Streamlit**
- **joblib**

---

## 📂 Project Structure

movie-recommender-ui/
│
├── data/
│ ├── ratings.csv
│ └── movies.csv
│
├── models/
│ ├── svd_model.pkl
│ ├── tfidf.pkl
│ └── cosine_sim.pkl
│
├── train.py # Offline model training
├── app.py # Streamlit UI (inference only)
├── requirements.txt
└── README.md

---

## 📊 Dataset

- **MovieLens Dataset (ml-latest-small)**
- Contains movie metadata and user ratings
- Source: GroupLens Research

---

## ⚙️ Setup Instructions

```bash
1️⃣ Clone the repository
git clone https://github.com/<your-username>/<repo-name>.git
cd movie-recommender-ui
2️⃣ Create and activate virtual environment
python -m venv venv
venv\Scripts\activate   # Windows

3️⃣ Install dependencies
pip install -r requirements.txt

🧪 Train the Models (Run Once)
python train.py

This will:
-Train the content-based model
-Train the collaborative filtering model (SVD)
-Save trained models to the models/ folder

▶️ Run the Streamlit App
streamlit run app.py
Open browser at:
http://localhost:8501

🧠 Recommendation Approaches
🔹 Content-Based Filtering
-Uses movie genres
-TF-IDF vectorization
-Cosine similarity to find similar movies

🔹 Collaborative Filtering
-Learns user–item interactions
-Matrix factorization using SVD
-Predicts ratings for unseen movies

🔹 Hybrid Recommendation
-Combines content similarity and collaborative predictions
-Weighted scoring to improve personalization and cold-start handling

📈 Evaluation
-Model evaluated using RMSE
-Achieved RMSE ≈ 0.87 on MovieLens dataset

🌱 Future Improvements
-Add movie posters using TMDB API
-Add alpha slider for hybrid weighting
-Precision@K / Recall@K evaluation
-User login & personalization
-Cloud deployment (Streamlit Cloud / Render)

👤 Author
Rhithikaa Ramkumar
B.Tech CSE Student
Exploring Machine Learning & Software Development 🚀

⭐ Acknowledgements
GroupLens Research
Surprise library
Streamlit community
