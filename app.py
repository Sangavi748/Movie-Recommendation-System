from flask import Flask, request, jsonify, render_template_string
import pandas as pd
import numpy as np
import os
import sqlite3
import kagglehub
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

DATASET = "grouplens/movielens-20m-dataset"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_FILE = os.path.join(BASE_DIR, "movie_recommendation.db")
DATA_DIR = os.path.join(BASE_DIR, "data")

os.makedirs(DATA_DIR, exist_ok=True)

def find_file(root, filename):
    if not root or not os.path.exists(root):
        return None
    for current_root, dirs, files in os.walk(root):
        for file in files:
            if file.lower() == filename.lower():
                return os.path.join(current_root, file)
    return None

def load_dataset():
    movie_path = find_file(DATA_DIR, "movie.csv")
    rating_path = find_file(DATA_DIR, "rating.csv")

    cache_root = os.path.expanduser(
        r"~\.cache\kagglehub\datasets\grouplens\movielens-20m-dataset"
    )

    if movie_path is None:
        movie_path = find_file(cache_root, "movie.csv")

    if rating_path is None:
        rating_path = find_file(cache_root, "rating.csv")

    if movie_path is None or rating_path is None:
        downloaded_path = kagglehub.dataset_download(DATASET)

        movie_path = find_file(downloaded_path, "movie.csv")
        rating_path = find_file(downloaded_path, "rating.csv")

        if movie_path is None:
            movie_path = find_file(cache_root, "movie.csv")

        if rating_path is None:
            rating_path = find_file(cache_root, "rating.csv")

    if movie_path is None or rating_path is None:
        raise FileNotFoundError(
            "Kaggle dataset was found, but movie.csv or rating.csv could not be located."
        )

    movies = pd.read_csv(movie_path)
    ratings = pd.read_csv(rating_path, nrows=1000000)

    movies.columns = [str(c).strip() for c in movies.columns]
    ratings.columns = [str(c).strip() for c in ratings.columns]

    if "movieId" not in movies.columns:
        if "movie_id" in movies.columns:
            movies = movies.rename(columns={"movie_id": "movieId"})

    if "movieId" not in ratings.columns:
        if "movie_id" in ratings.columns:
            ratings = ratings.rename(columns={"movie_id": "movieId"})

    if "userId" not in ratings.columns:
        if "user_id" in ratings.columns:
            ratings = ratings.rename(columns={"user_id": "userId"})

    if "movieId" not in movies.columns:
        raise ValueError("movie.csv does not contain movieId column.")

    if "title" not in movies.columns:
        raise ValueError("movie.csv does not contain title column.")

    if "genres" not in movies.columns:
        movies["genres"] = ""

    movies["movieId"] = pd.to_numeric(movies["movieId"], errors="coerce")
    ratings["movieId"] = pd.to_numeric(ratings["movieId"], errors="coerce")
    ratings["rating"] = pd.to_numeric(ratings["rating"], errors="coerce")

    movies = movies.dropna(subset=["movieId"]).copy()
    ratings = ratings.dropna(subset=["movieId", "rating"]).copy()

    movies["movieId"] = movies["movieId"].astype(int)
    ratings["movieId"] = ratings["movieId"].astype(int)

    movies["title"] = movies["title"].fillna("").astype(str)
    movies["genres"] = movies["genres"].fillna("").astype(str)

    movies = movies.drop_duplicates("movieId").reset_index(drop=True)

    return movies, ratings

movies, ratings = load_dataset()

movies["content"] = (
    movies["title"].fillna("")
    + " "
    + movies["genres"].fillna("").str.replace("|", " ", regex=False)
)

tfidf = TfidfVectorizer(
    stop_words="english",
    max_features=20000
)

tfidf_matrix = tfidf.fit_transform(movies["content"])

movie_id_to_index = {
    int(movie_id): index
    for index, movie_id in enumerate(movies["movieId"])
}

def initialize_database():
    connection = sqlite3.connect(DB_FILE)

    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            user_id TEXT PRIMARY KEY,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS user_ratings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            movie_id INTEGER,
            rating REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS recommendation_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            movie_id INTEGER,
            score REAL,
            recommendation_type TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    movies[["movieId", "title", "genres"]].to_sql(
        "movies",
        connection,
        if_exists="replace",
        index=False
    )

    connection.commit()
    connection.close()

initialize_database()

def create_user(user_id):
    connection = sqlite3.connect(DB_FILE)

    connection.execute(
        "INSERT OR IGNORE INTO users (user_id) VALUES (?)",
        (user_id,)
    )

    connection.commit()
    connection.close()

def get_movie(movie_id):
    result = movies[movies["movieId"] == int(movie_id)]

    if result.empty:
        return None

    row = result.iloc[0]

    return {
        "movieId": int(row["movieId"]),
        "title": row["title"],
        "genres": row["genres"]
    }

def content_recommendations(movie_id, limit=10):
    movie_id = int(movie_id)

    if movie_id not in movie_id_to_index:
        return []

    index = movie_id_to_index[movie_id]

    similarities = cosine_similarity(
        tfidf_matrix[index:index + 1],
        tfidf_matrix
    ).flatten()

    similar_indices = similarities.argsort()[::-1]

    results = []

    for item_index in similar_indices:
        candidate_id = int(movies.iloc[item_index]["movieId"])

        if candidate_id == movie_id:
            continue

        results.append(
            {
                "movieId": candidate_id,
                "title": movies.iloc[item_index]["title"],
                "genres": movies.iloc[item_index]["genres"],
                "score": round(float(similarities[item_index]), 4),
                "type": "Content Based"
            }
        )

        if len(results) >= limit:
            break

    return results

def collaborative_recommendations(movie_id, limit=10):
    movie_id = int(movie_id)

    related_users = ratings[
        (ratings["movieId"] == movie_id)
        & (ratings["rating"] >= 4)
    ]["userId"].unique()

    if len(related_users) == 0:
        popular = (
            ratings[ratings["movieId"] != movie_id]
            .groupby("movieId")
            .agg(
                avg_rating=("rating", "mean"),
                rating_count=("rating", "count")
            )
            .reset_index()
        )

        popular["score"] = (
            popular["avg_rating"] *
            np.log1p(popular["rating_count"])
        )

        popular = popular.sort_values(
            "score",
            ascending=False
        ).head(limit)

        result = []

        for _, row in popular.iterrows():
            movie = get_movie(int(row["movieId"]))

            if movie:
                result.append(
                    {
                        "movieId": movie["movieId"],
                        "title": movie["title"],
                        "genres": movie["genres"],
                        "score": round(float(row["avg_rating"]) / 5, 4),
                        "type": "Collaborative"
                    }
                )

        return result

    candidate_ratings = ratings[
        ratings["userId"].isin(related_users)
        & (ratings["movieId"] != movie_id)
    ]

    if candidate_ratings.empty:
        return []

    grouped = (
        candidate_ratings
        .groupby("movieId")
        .agg(
            avg_rating=("rating", "mean"),
            rating_count=("rating", "count")
        )
        .reset_index()
    )

    grouped["score"] = (
        grouped["avg_rating"] / 5
        * np.log1p(grouped["rating_count"])
    )

    grouped = grouped.sort_values(
        "score",
        ascending=False
    ).head(limit)

    result = []

    for _, row in grouped.iterrows():
        movie = get_movie(int(row["movieId"]))

        if movie:
            result.append(
                {
                    "movieId": movie["movieId"],
                    "title": movie["title"],
                    "genres": movie["genres"],
                    "score": round(
                        min(float(row["avg_rating"]) / 5, 1),
                        4
                    ),
                    "type": "Collaborative"
                }
            )

    return result

def hybrid_recommendations(movie_id, limit=10):
    content = content_recommendations(movie_id, 20)
    collaborative = collaborative_recommendations(movie_id, 20)

    combined = {}

    for item in content:
        combined[item["movieId"]] = {
            "movieId": item["movieId"],
            "title": item["title"],
            "genres": item["genres"],
            "content_score": item["score"],
            "collaborative_score": 0,
            "type": "Hybrid"
        }

    for item in collaborative:
        if item["movieId"] not in combined:
            combined[item["movieId"]] = {
                "movieId": item["movieId"],
                "title": item["title"],
                "genres": item["genres"],
                "content_score": 0,
                "collaborative_score": item["score"],
                "type": "Hybrid"
            }
        else:
            combined[item["movieId"]]["collaborative_score"] = item["score"]

    result = []

    for item in combined.values():
        item["score"] = round(
            0.6 * item["content_score"]
            + 0.4 * item["collaborative_score"],
            4
        )

        result.append(item)

    result.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    return result[:limit]

def popular_movies(limit=20):
    stats = (
        ratings
        .groupby("movieId")
        .agg(
            avg_rating=("rating", "mean"),
            rating_count=("rating", "count")
        )
        .reset_index()
    )

    stats["score"] = (
        stats["avg_rating"] *
        np.log1p(stats["rating_count"])
    )

    stats = stats.sort_values(
        "score",
        ascending=False
    ).head(limit)

    result = []

    for _, row in stats.iterrows():
        movie = get_movie(int(row["movieId"]))

        if movie:
            result.append(
                {
                    "movieId": movie["movieId"],
                    "title": movie["title"],
                    "genres": movie["genres"],
                    "rating": round(float(row["avg_rating"]), 2),
                    "rating_count": int(row["rating_count"])
                }
            )

    return result

@app.route("/")
def home():
    return render_template_string(HTML_PAGE)

@app.route("/api/search")
def search_movies():
    query = request.args.get("q", "").strip()
    genre = request.args.get("genre", "").strip()

    result = movies.copy()

    if query:
        result = result[
            result["title"].str.contains(
                query,
                case=False,
                na=False
            )
        ]

    if genre:
        result = result[
            result["genres"].str.contains(
                genre,
                case=False,
                na=False
            )
        ]

    result = result.head(30)

    return jsonify(
        [
            {
                "movieId": int(row["movieId"]),
                "title": row["title"],
                "genres": row["genres"]
            }
            for _, row in result.iterrows()
        ]
    )

@app.route("/api/movie/<int:movie_id>")
def movie_details(movie_id):
    movie = get_movie(movie_id)

    if movie is None:
        return jsonify({"error": "Movie not found"}), 404

    movie_ratings = ratings[
        ratings["movieId"] == movie_id
    ]["rating"]

    movie["average_rating"] = (
        round(float(movie_ratings.mean()), 2)
        if not movie_ratings.empty
        else 0
    )

    movie["rating_count"] = int(
        movie_ratings.count()
    )

    return jsonify(movie)

@app.route("/api/recommend/<int:movie_id>")
def recommend(movie_id):
    recommendation_type = request.args.get(
        "type",
        "hybrid"
    ).lower()

    limit = int(
        request.args.get(
            "limit",
            10
        )
    )

    if recommendation_type == "content":
        result = content_recommendations(
            movie_id,
            limit
        )
    elif recommendation_type == "collaborative":
        result = collaborative_recommendations(
            movie_id,
            limit
        )
    else:
        result = hybrid_recommendations(
            movie_id,
            limit
        )

    return jsonify(result)

@app.route("/api/popular")
def popular():
    limit = int(
        request.args.get(
            "limit",
            20
        )
    )

    return jsonify(
        popular_movies(limit)
    )

@app.route("/api/rate", methods=["POST"])
def rate_movie():
    data = request.get_json()

    user_id = str(
        data.get(
            "user_id",
            "guest"
        )
    ).strip()

    movie_id = int(
        data.get("movie_id")
    )

    rating_value = float(
        data.get("rating")
    )

    if not user_id:
        user_id = "guest"

    if rating_value < 0.5 or rating_value > 5:
        return jsonify(
            {
                "error": "Rating must be between 0.5 and 5"
            }
        ), 400

    if get_movie(movie_id) is None:
        return jsonify(
            {
                "error": "Movie not found"
            }
        ), 404

    create_user(user_id)

    connection = sqlite3.connect(DB_FILE)

    connection.execute(
        """
        DELETE FROM user_ratings
        WHERE user_id = ? AND movie_id = ?
        """,
        (
            user_id,
            movie_id
        )
    )

    connection.execute(
        """
        INSERT INTO user_ratings
        (user_id, movie_id, rating)
        VALUES (?, ?, ?)
        """,
        (
            user_id,
            movie_id,
            rating_value
        )
    )

    connection.commit()
    connection.close()

    return jsonify(
        {
            "message": "Rating saved successfully"
        }
    )

@app.route("/api/history")
def history():
    user_id = request.args.get(
        "user_id",
        "guest"
    )

    connection = sqlite3.connect(DB_FILE)

    query = """
        SELECT
            recommendation_history.id,
            recommendation_history.movie_id,
            recommendation_history.score,
            recommendation_history.recommendation_type,
            recommendation_history.created_at,
            movies.title,
            movies.genres
        FROM recommendation_history
        LEFT JOIN movies
        ON recommendation_history.movie_id = movies.movieId
        WHERE recommendation_history.user_id = ?
        ORDER BY recommendation_history.created_at DESC
        LIMIT 50
    """

    result = pd.read_sql_query(
        query,
        connection,
        params=(user_id,)
    )

    connection.close()

    return jsonify(
        result.to_dict(
            orient="records"
        )
    )

@app.route("/api/save-history", methods=["POST"])
def save_history():
    data = request.get_json()

    user_id = str(
        data.get(
            "user_id",
            "guest"
        )
    )

    movie_id = int(
        data.get("movie_id")
    )

    score = float(
        data.get(
            "score",
            0
        )
    )

    recommendation_type = str(
        data.get(
            "type",
            "Hybrid"
        )
    )

    create_user(user_id)

    connection = sqlite3.connect(DB_FILE)

    connection.execute(
        """
        INSERT INTO recommendation_history
        (user_id, movie_id, score, recommendation_type)
        VALUES (?, ?, ?, ?)
        """,
        (
            user_id,
            movie_id,
            score,
            recommendation_type
        )
    )

    connection.commit()
    connection.close()

    return jsonify(
        {
            "message": "Recommendation history saved"
        }
    )

@app.route("/api/dashboard")
def dashboard():
    connection = sqlite3.connect(DB_FILE)

    users_count = connection.execute(
        "SELECT COUNT(*) FROM users"
    ).fetchone()[0]

    user_ratings_count = connection.execute(
        "SELECT COUNT(*) FROM user_ratings"
    ).fetchone()[0]

    history_count = connection.execute(
        "SELECT COUNT(*) FROM recommendation_history"
    ).fetchone()[0]

    connection.close()

    return jsonify(
        {
            "movies": int(len(movies)),
            "ratings_used": int(len(ratings)),
            "users": int(users_count),
            "user_ratings": int(user_ratings_count),
            "recommendations": int(history_count)
        }
    )

HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Movie Recommendation System</title>
<style>
* {
    box-sizing: border-box;
}

body {
    margin: 0;
    font-family: Arial, sans-serif;
    background: #f4f6f8;
    color: #222;
}

header {
    background: #111827;
    color: white;
    padding: 22px;
    text-align: center;
}

header h1 {
    margin: 0 0 8px;
}

header p {
    margin: 0;
}

.container {
    width: 92%;
    max-width: 1250px;
    margin: 25px auto;
}

.panel {
    background: white;
    padding: 22px;
    border-radius: 12px;
    margin-bottom: 22px;
    box-shadow: 0 3px 12px rgba(0,0,0,0.08);
}

.controls {
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
}

input, select, button {
    padding: 12px;
    border-radius: 8px;
    border: 1px solid #ccc;
    font-size: 15px;
}

input {
    flex: 1;
    min-width: 220px;
}

button {
    background: #111827;
    color: white;
    cursor: pointer;
    border: none;
}

button:hover {
    opacity: 0.85;
}

.movie-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    gap: 18px;
}

.movie-card {
    background: white;
    border: 1px solid #ddd;
    border-radius: 12px;
    padding: 18px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}

.movie-card h3 {
    margin-top: 0;
}

.genre {
    color: #666;
    font-size: 14px;
    min-height: 40px;
}

.score {
    font-weight: bold;
    margin: 10px 0;
}

.rating-buttons {
    display: flex;
    gap: 4px;
    flex-wrap: wrap;
    margin-top: 10px;
}

.rating-buttons button {
    padding: 7px 10px;
    background: #374151;
}

.stats {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 15px;
}

.stat {
    background: #111827;
    color: white;
    padding: 20px;
    border-radius: 10px;
    text-align: center;
}

.stat h2 {
    margin: 0;
}

.stat p {
    margin-bottom: 0;
}

.message {
    padding: 12px;
    background: #e5e7eb;
    border-radius: 8px;
    margin-top: 12px;
}

.hidden {
    display: none;
}

.section-title {
    margin-top: 0;
}

.small {
    font-size: 13px;
    color: #666;
}
</style>
</head>

<body>

<header>
<h1>Movie Recommendation System</h1>
<p>AI-Based Movie Search and Recommendation Platform</p>
</header>

<div class="container">

<div class="panel">
<h2 class="section-title">User</h2>

<div class="controls">
<input id="userId" value="guest" placeholder="Enter User ID">
</div>

<p class="small">
Enter a user ID to store ratings and recommendation history.
</p>
</div>

<div class="panel">
<h2 class="section-title">Movie Search</h2>

<div class="controls">
<input id="searchBox" placeholder="Search movie name">
<select id="genre">
<option value="">All Genres</option>
<option value="Action">Action</option>
<option value="Adventure">Adventure</option>
<option value="Animation">Animation</option>
<option value="Children">Children</option>
<option value="Comedy">Comedy</option>
<option value="Crime">Crime</option>
<option value="Documentary">Documentary</option>
<option value="Drama">Drama</option>
<option value="Fantasy">Fantasy</option>
<option value="Horror">Horror</option>
<option value="Mystery">Mystery</option>
<option value="Romance">Romance</option>
<option value="Sci-Fi">Sci-Fi</option>
<option value="Thriller">Thriller</option>
<option value="War">War</option>
<option value="Western">Western</option>
</select>
<button onclick="searchMovies()">Search</button>
</div>

<div id="searchResults" class="movie-grid" style="margin-top:20px;"></div>
</div>

<div class="panel">
<h2 class="section-title">Recommendations</h2>

<div class="controls">
<button onclick="loadPopular()">Popular Movies</button>
<button onclick="setRecommendationType('content')">Content Based</button>
<button onclick="setRecommendationType('collaborative')">Collaborative</button>
<button onclick="setRecommendationType('hybrid')">Hybrid</button>
</div>

<div id="selectedMovie" class="message">
Select a movie to get recommendations.
</div>

<div id="recommendations" class="movie-grid" style="margin-top:20px;"></div>
</div>

<div class="panel">
<h2 class="section-title">Dashboard</h2>

<div class="stats">
<div class="stat">
<h2 id="moviesCount">0</h2>
<p>Movies</p>
</div>

<div class="stat">
<h2 id="ratingsCount">0</h2>
<p>Ratings Used</p>
</div>

<div class="stat">
<h2 id="usersCount">0</h2>
<p>Users</p>
</div>

<div class="stat">
<h2 id="userRatingsCount">0</h2>
<p>User Ratings</p>
</div>

<div class="stat">
<h2 id="recommendationsCount">0</h2>
<p>Recommendations</p>
</div>
</div>
</div>

<div class="panel">
<h2 class="section-title">Recommendation History</h2>
<button onclick="loadHistory()">Load History</button>
<div id="history" class="movie-grid" style="margin-top:20px;"></div>
</div>

</div>

<script>

let recommendationType = "hybrid";
let selectedMovieId = null;

async function searchMovies() {
    const query = document.getElementById("searchBox").value;
    const genre = document.getElementById("genre").value;

    const response = await fetch(
        "/api/search?q=" +
        encodeURIComponent(query) +
        "&genre=" +
        encodeURIComponent(genre)
    );

    const data = await response.json();

    displayMovies(
        data,
        document.getElementById("searchResults"),
        false
    );
}

function displayMovies(data, container, recommendationMode) {
    container.innerHTML = "";

    if (data.length === 0) {
        container.innerHTML = "<p>No movies found.</p>";
        return;
    }

    data.forEach(movie => {
        const card = document.createElement("div");
        card.className = "movie-card";

        let extra = "";

        if (movie.score !== undefined) {
            extra += "<div class='score'>Recommendation Score: " +
                movie.score +
                "</div>";
        }

        if (movie.rating !== undefined) {
            extra += "<div class='score'>Average Rating: " +
                movie.rating +
                "</div>";
        }

        if (movie.rating_count !== undefined) {
            extra += "<div class='small'>Ratings: " +
                movie.rating_count +
                "</div>";
        }

        card.innerHTML =
            "<h3>" +
            escapeHtml(movie.title) +
            "</h3>" +
            "<div class='genre'>" +
            escapeHtml(movie.genres) +
            "</div>" +
            extra +
            "<button onclick='getRecommendations(" +
            movie.movieId +
            ", \"" +
            escapeHtml(movie.title).replace(/"/g, "&quot;") +
            "\")'>Recommend Similar</button>" +
            "<div class='rating-buttons'>" +
            "<button onclick='rateMovie(" + movie.movieId + ", 1)'>1</button>" +
            "<button onclick='rateMovie(" + movie.movieId + ", 2)'>2</button>" +
            "<button onclick='rateMovie(" + movie.movieId + ", 3)'>3</button>" +
            "<button onclick='rateMovie(" + movie.movieId + ", 4)'>4</button>" +
            "<button onclick='rateMovie(" + movie.movieId + ", 5)'>5</button>" +
            "</div>";

        container.appendChild(card);
    });
}

async function getRecommendations(movieId, title) {
    selectedMovieId = movieId;

    document.getElementById("selectedMovie").innerHTML =
        "Selected movie: <strong>" +
        escapeHtml(title) +
        "</strong>";

    const response = await fetch(
        "/api/recommend/" +
        movieId +
        "?type=" +
        recommendationType +
        "&limit=12"
    );

    const data = await response.json();

    displayMovies(
        data,
        document.getElementById("recommendations"),
        true
    );

    const userId =
        document.getElementById("userId").value || "guest";

    for (const movie of data) {
        await fetch("/api/save-history", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                user_id: userId,
                movie_id: movie.movieId,
                score: movie.score || 0,
                type: recommendationType
            })
        });
    }

    loadDashboard();
}

function setRecommendationType(type) {
    recommendationType = type;

    if (selectedMovieId !== null) {
        getRecommendations(
            selectedMovieId,
            "Selected Movie"
        );
    }
}

async function loadPopular() {
    const response = await fetch(
        "/api/popular?limit=20"
    );

    const data = await response.json();

    displayMovies(
        data,
        document.getElementById("recommendations"),
        false
    );

    document.getElementById("selectedMovie").innerHTML =
        "Popular movies from the MovieLens dataset.";
}

async function rateMovie(movieId, rating) {
    const userId =
        document.getElementById("userId").value || "guest";

    const response = await fetch(
        "/api/rate",
        {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                user_id: userId,
                movie_id: movieId,
                rating: rating
            })
        }
    );

    const data = await response.json();

    document.getElementById("selectedMovie").innerHTML =
        data.message ||
        data.error ||
        "Rating saved.";

    loadDashboard();
}

async function loadDashboard() {
    const response = await fetch(
        "/api/dashboard"
    );

    const data = await response.json();

    document.getElementById("moviesCount").innerText =
        data.movies;

    document.getElementById("ratingsCount").innerText =
        data.ratings_used;

    document.getElementById("usersCount").innerText =
        data.users;

    document.getElementById("userRatingsCount").innerText =
        data.user_ratings;

    document.getElementById("recommendationsCount").innerText =
        data.recommendations;
}

async function loadHistory() {
    const userId =
        document.getElementById("userId").value || "guest";

    const response = await fetch(
        "/api/history?user_id=" +
        encodeURIComponent(userId)
    );

    const data = await response.json();

    const container =
        document.getElementById("history");

    container.innerHTML = "";

    if (data.length === 0) {
        container.innerHTML =
            "<p>No recommendation history found.</p>";
        return;
    }

    data.forEach(item => {
        const card =
            document.createElement("div");

        card.className = "movie-card";

        card.innerHTML =
            "<h3>" +
            escapeHtml(item.title || "Unknown Movie") +
            "</h3>" +
            "<div class='genre'>" +
            escapeHtml(item.genres || "") +
            "</div>" +
            "<div class='score'>Score: " +
            item.score +
            "</div>" +
            "<div class='small'>Type: " +
            escapeHtml(item.recommendation_type) +
            "</div>";

        container.appendChild(card);
    });
}

function escapeHtml(value) {
    return String(value)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}

loadDashboard();

</script>

</body>
</html>
"""

if __name__ == "__main__":
    print("Movie Recommendation System")
    print("Movies loaded:", len(movies))
    print("Ratings loaded:", len(ratings))
    print("Starting Flask server...")
    print("Open http://127.0.0.1:5000 in your browser")
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )