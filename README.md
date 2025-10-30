# Movie-Recommender-System
# 🎬 Movie Recommender System (Flask + TMDB API)

A web-based Movie Recommendation System built with **Flask**, **TMDB**, and **IMDb** data.  
It suggests similar movies based on user input and displays posters using the TMDB API.  
The project uses **content-based filtering** with cosine similarity to recommend movies.

---

## 🚀 Features

- 🔍 Search for any movie by title  
- 🎥 Get top similar movie recommendations  
- 🖼️ Fetch and display movie posters dynamically via TMDB API  
- 💡 Responsive and modern dark-themed UI  
- ⚡ Fast and simple Flask backend with Pandas & Scikit-learn  
- 🧩 Secure API key handling using `.env` and environment variables  

---

## 🧠 Tech Stack

**Backend:** Flask, Python, Pandas, NumPy, Scikit-learn  
**Frontend:** HTML, CSS, JavaScript  
**API:** TMDB (via RapidAPI or TMDB direct API)  
**Data:** TMDB 5000 Movies + Credits Dataset  

---

## 📂 Project Structure

movie-recommender/
│
├── app.py # Flask main app
├── tmdb_5000_movies.csv # Dataset
├── tmdb_5000_credits.csv # Dataset
├── static/ # CSS, JS, and assets
├── templates/ # HTML files (index.html, recommend.html, etc.)
├── .env # API key (not uploaded)
├── .gitignore
└── README.md


---

## ⚙️ Setup & Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/movie-recommender.git
   cd movie-recommender


2. **Create a virtual environment**

python -m venv venv
source venv/bin/activate      # on macOS/Linux
venv\Scripts\activate         # on Windows


3. **Install dependencies**

pip install -r requirements.txt


4. **Add your TMDB API key**
Create a file named .env in the root folder:

TMDB_API_KEY=your_api_key_here


5. **Run the Flask app**

python app.py


6. **Open in browser**

http://127.0.0.1:5000/


## 🌐 Deployment

You can deploy this Flask app on:

Render

Railway

Vercel (with serverless functions)

Heroku (legacy support)

Be sure to set your TMDB_API_KEY in the environment settings of your deployment platform.

## 🧾 License

This project is open-source and available under the MIT License.

## 💬 Contact

Developed by Tarnvir Singh
📧 Email: tarnsaini9713@gmail.com

🌐 GitHub:  Taran-Saini13

⭐ If you like this project, consider giving it a star!