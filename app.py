# app.py
import ast
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem.porter import PorterStemmer
import requests
import os
from dotenv import load_dotenv
load_dotenv()

# ---------- CONFIG ----------
TMDB_API_KEY = os.getenv("TMDB_API_KEY")  # <-- replace with your TMDB key or keep placeholder
TMDB_SEARCH_URL = "https://api.themoviedb.org/3/search/movie"
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w342"  # poster size w342 is a good balance

# ---------- HELPER FUNCTIONS ----------
ps = PorterStemmer()

def stem_text(text: str) -> str:
    # simple word-by-word Porter stemming
    return " ".join(ps.stem(w) for w in str(text).split())

def safe_list_to_names(obj):
    """Convert JSON-like string to list of 'name' values."""
    try:
        parsed = ast.literal_eval(obj)
    except Exception:
        return []
    names = []
    for item in parsed:
        name = item.get("name")
        if name:
            names.append(name.replace(" ", ""))  # remove internal spaces
    return names

def take_top_cast(obj, top_n=3):
    try:
        parsed = ast.literal_eval(obj)
    except Exception:
        return []
    names = []
    for i, item in enumerate(parsed):
        if i >= top_n:
            break
        name = item.get("name")
        if name:
            names.append(name.replace(" ", ""))
    return names

def fetch_director(obj):
    try:
        parsed = ast.literal_eval(obj)
    except Exception:
        return []
    for item in parsed:
        if item.get("job", "").lower() == "director":
            name = item.get("name", "")
            return [name.replace(" ", "")] if name else []
    return []

def fetch_poster_tmdb(movie_title: str):
    """Search TMDB for movie_title and return full poster url or placeholder."""
    if not TMDB_API_KEY or "YOUR_TMDB_API_KEY" in TMDB_API_KEY:
        # placeholder if api key not provided
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
        # network error or JSON parse error -> placeholder
        print("TMDB fetch error:", e)
        return "https://via.placeholder.com/300x450?text=No+Image"

# ---------- LOAD & PREPROCESS DATA (done once at startup) ----------
print("Loading datasets...")
movies_df = pd.read_csv("tmdb_5000_movies.csv")
credits_df = pd.read_csv("tmdb_5000_credits.csv")

# merge
movies = movies_df.merge(credits_df, on="title")

# keep columns we care about
movies = movies[["id", "title", "overview", "genres", "keywords", "cast", "crew"]].dropna(subset=["overview"])

# convert fields
movies["genres"] = movies["genres"].apply(safe_list_to_names)
movies["keywords"] = movies["keywords"].apply(safe_list_to_names)
movies["cast"] = movies["cast"].apply(lambda x: take_top_cast(x, top_n=3))
movies["crew"] = movies["crew"].apply(fetch_director)
movies["overview"] = movies["overview"].apply(lambda x: str(x).split())  # split overview into words

# remove spaces within tokens (we already did for cast/crew/genres above)
# combine tags
movies["tags"] = movies["cast"] + movies["crew"] + movies["genres"] + movies["keywords"] + movies["overview"]

# new_df to use
new_df = movies[["id", "title", "tags"]].copy()
# convert tags list to string
new_df["tags"] = new_df["tags"].apply(lambda x: " ".join(x))
# lower + stemming
new_df["tags"] = new_df["tags"].str.lower().apply(stem_text)
# add a lowercase title column for quick matching
new_df["title_lower"] = new_df["title"].str.lower()

# Vectorize and compute similarity
print("Vectorizing and computing similarity matrix...")
cv = CountVectorizer(max_features=5000, stop_words="english")
vectors = cv.fit_transform(new_df["tags"]).toarray()
similarity = cosine_similarity(vectors)

# Build a set or list of titles for suggestions (fast)
titles_list = new_df["title"].tolist()
titles_lower = new_df["title_lower"].tolist()

print("Startup complete. Movie count:", len(new_df))

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
        # simple prefix match on title_lower
        # return up to 7 suggestions sorted by closeness (startsWith first then contains)
        starts = [t for t in titles_list if t.lower().startswith(q)]
        contains = [t for t in titles_list if q in t.lower() and not t.lower().startswith(q)]
        combined = starts + contains
        suggestions = combined[:7]
    return jsonify(suggestions)

def recommend(movie_name: str, top_n=6):
    movie_name = movie_name.strip().lower()
    results = []
    # find best match index
    try:
        idx = new_df[new_df["title_lower"] == movie_name].index[0]
    except Exception:
        # fallback: try to find best approximate by contains
        matches = new_df[new_df["title_lower"].str.contains(movie_name)]
        if matches.shape[0] > 0:
            idx = matches.index[0]
        else:
            # not found
            return [{"title": "Movie not found", "poster": "", "note": f"No movie matching '{movie_name}'"}]

    distances = similarity[idx]
    movies_list = sorted(list(enumerate(distances)), key=lambda x: x[1], reverse=True)[1:top_n]
    for i, score in movies_list:
        title = new_df.iloc[i]["title"]
        poster = fetch_poster_tmdb(title)
        results.append({"title": title, "poster": poster})
    return results

if __name__ == "__main__":
    # set host=0.0.0.0 if you want to serve externally
    app.run(debug=True)
