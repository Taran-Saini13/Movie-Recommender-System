# app.py
import ast
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests
import os
from dotenv import load_dotenv

load_dotenv()

# ---------- CONFIG ----------
TMDB_API_KEY = os.getenv("TMDB_API_KEY")
TMDB_SEARCH_URL = "https://api.themoviedb.org/3/search/movie"
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w342"  # good size for posters

# ---------- HELPER FUNCTIONS ----------
ps = PorterStemmer()


def stem_text(text: str) -> str:
    return " ".join(ps.stem(w) for w in str(text).split())


def fetch_poster_tmdb(movie_title: str):
    """Search TMDB for movie_title and return full poster url or placeholder."""
    if not TMDB_API_KEY or "YOUR_TMDB_API_KEY" in TMDB_API_KEY:
        return "https://via.placeholder.com/300x450?text=No+Image"

    params = {
        "api_key": TMDB_API_KEY,
        "query": movie_title,
        "include_adult": "false",
        "page": 1,
    }
    try:
        r = requests.get(TMDB_SEARCH_URL, params=params, timeout=5)
        r.raise_for_status()
        data = r.json()
        results = data.get("results", [])
        if results:
            poster_path = results[0].get("poster_path")
            if poster_path:
                return TMDB_IMAGE_BASE + poster_path
        return "https://via.placeholder.com/300x450?text=No+Image"
    except Exception as e:
        print("TMDB fetch error:", e)
        return "https://via.placeholder.com/300x450?text=No+Image"


# ---------- LOAD PREPROCESSED DATA ----------
print("Loading preprocessed data...")
new_df = pd.read_csv("processed_movies.csv")
similarity = np.load("similarity.npy")

titles_list = new_df["title"].tolist()
titles_lower = [t.lower() for t in titles_list]
print("✅ Preprocessed data loaded. Movie count:", len(new_df))


# ---------- FLASK APP ----------
app = Flask(__name__, static_folder="static", template_folder="templates")


@app.route("/", methods=["GET", "POST"])
def home():
    recommended = []
    query_movie = ""
    if request.method == "POST":
        query_movie = request.form.get("movie", "").strip()
        if query_movie:
            recommended = recommend(query_movie)
    return render_template("index.html", recommended_movies=recommended)


@app.route("/suggest")
def suggest():
    q = request.args.get("q", "").strip().lower()
    suggestions = []
    if q:
        starts = [t for t in titles_list if t.lower().startswith(q)]
        contains = [
            t for t in titles_list if q in t.lower() and not t.lower().startswith(q)
        ]
        combined = starts + contains
        suggestions = combined[:7]
    return jsonify(suggestions)


# ---------- RECOMMENDATION FUNCTION ----------
def recommend(movie_name: str, top_n=6):
    movie_name = movie_name.strip().lower()

    # build lowercase list dynamically
    titles_lower = [t.lower() for t in new_df["title"].tolist()]

    if movie_name not in titles_lower:
        matches = [t for t in titles_lower if movie_name in t]
        if not matches:
            return [
                {
                    "title": "Movie not found",
                    "poster": "https://via.placeholder.com/300x450?text=Not+Found",
                    "note": f"No movie matching '{movie_name}'",
                }
            ]
        movie_name = matches[0]

    idx = titles_lower.index(movie_name)
    distances = similarity[idx]
    movie_list = sorted(
        list(enumerate(distances)), reverse=True, key=lambda x: x[1]
    )[1 : top_n + 1]

    results = []
    for i, score in movie_list:
        title = new_df.iloc[i]["title"]
        poster = fetch_poster_tmdb(title)
        results.append({"title": title, "poster": poster})

    return results


# ---------- MAIN ----------
if __name__ == "__main__":
    app.run(debug=True)
