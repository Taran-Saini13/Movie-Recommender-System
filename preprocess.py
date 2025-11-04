import pandas as pd
import numpy as np
import ast
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ps = PorterStemmer()

def safe_list_to_names(obj):
    try:
        parsed = ast.literal_eval(obj)
    except Exception:
        return []
    names = [i.get("name", "").replace(" ", "") for i in parsed if "name" in i]
    return names

def take_top_cast(obj, top_n=3):
    try:
        parsed = ast.literal_eval(obj)
    except Exception:
        return []
    names = [i["name"].replace(" ", "") for i in parsed[:top_n] if "name" in i]
    return names

def fetch_director(obj):
    try:
        parsed = ast.literal_eval(obj)
    except Exception:
        return []
    for i in parsed:
        if i.get("job") == "Director":
            return [i["name"].replace(" ", "")]
    return []

def stem_text(text):
    return " ".join(ps.stem(w) for w in text.split())

print("Loading CSVs...")
movies = pd.read_csv("tmdb_5000_movies.csv")
credits = pd.read_csv("tmdb_5000_credits.csv")

movies = movies.merge(credits, on="title")
movies = movies[["id", "title", "overview", "genres", "keywords", "cast", "crew"]].dropna(subset=["overview"])

movies["genres"] = movies["genres"].apply(safe_list_to_names)
movies["keywords"] = movies["keywords"].apply(safe_list_to_names)
movies["cast"] = movies["cast"].apply(lambda x: take_top_cast(x, 3))
movies["crew"] = movies["crew"].apply(fetch_director)

movies["tags"] = (
    movies["overview"].fillna("").apply(lambda x: x.split()) +
    movies["genres"] +
    movies["keywords"] +
    movies["cast"] +
    movies["crew"]
)
new_df = movies[["id", "title", "tags"]].copy()
new_df["tags"] = new_df["tags"].apply(lambda x: " ".join(x).lower())
new_df["tags"] = new_df["tags"].apply(stem_text)

# optional: limit dataset size (1000–2000 movies)
new_df = new_df.head(1500)

print("Vectorizing...")
cv = CountVectorizer(max_features=5000, stop_words="english")
vectors = cv.fit_transform(new_df["tags"]).toarray()
similarity = cosine_similarity(vectors)

# Save everything
new_df.to_csv("processed_movies.csv", index=False)
np.save("similarity.npy", similarity)

print("✅ Saved processed_movies.csv and similarity.npy successfully.")
