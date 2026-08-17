import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from difflib import get_close_matches  # Galat spelling handle karne ke liye

# 1. Movie Dataset (Embedded List)
movies_data = [
    {"title": "Avatar", "genre": "Action Adventure Sci-Fi", "overview": "A paraplegic Marine dispatched to the moon Pandora on a unique mission."},
    {"title": "The Dark Knight", "genre": "Action Crime Drama", "overview": "When the menace known as the Joker wreaks havoc and chaos on Gotham."},
    {"title": "Inception", "genre": "Action Adventure Sci-Fi", "overview": "A thief who steals corporate secrets through the use of dream-sharing technology."},
    {"title": "Interstellar", "genre": "Adventure Drama Sci-Fi", "overview": "A team of explorers travel through a wormhole in space in an attempt to ensure humanity survival."},
    {"title": "The Avengers", "genre": "Action Sci-Fi", "overview": "Earth mightiest heroes must come together and learn to fight as a team to stop Loki."},
    {"title": "Iron Man", "genre": "Action Adventure Sci-Fi", "overview": "A billionaire engineer constructs an armored suit after being held captive in a cave."},
    {"title": "Titanic", "genre": "Drama Romance", "overview": "A seventeen-year-old aristocrat falls in love with a poor artist aboard the Titanic."},
    {"title": "The Notebook", "genre": "Drama Romance", "overview": "A young man and woman fall in love in the 1940s."},
    {"title": "La La Land", "genre": "Comedy Drama Music", "overview": "While navigating their careers in Los Angeles a pianist and an actress fall in love."},
    {"title": "The Godfather", "genre": "Crime Drama", "overview": "The aging patriarch of an organized crime dynasty transfers control to his son."},
    {"title": "Pulp Fiction", "genre": "Crime Drama", "overview": "The lives of two mob hitmen a boxer and a pair of diner bandits intertwine."},
    {"title": "Fight Club", "genre": "Drama", "overview": "An uninspired office worker and a soap maker form an underground fight club."}
]

# DataFrame banana
df = pd.DataFrame(movies_data)

# Genre aur Overview ko combine karke features tag banana
df['combined_features'] = df['genre'] + " " + df['overview']

# Lowercase titles fuzzy matching ke liye
df['title_lower'] = df['title'].str.lower()
all_titles_lower = df['title_lower'].tolist()

# 2. TF-IDF Vectorizer
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['combined_features'])

# 3. Cosine Similarity Matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# 4. Recommendation Function
def recommend_movies(movie_name):
    user_input = movie_name.strip().lower()

    # Step A: Exact match search karna
    matched = df[df['title_lower'] == user_input]

    # Step B: Partial match try karna (agar aadha naam likha ho)
    if matched.empty:
        matched = df[df['title_lower'].str.contains(user_input, na=False)]

    # Step C: Galat spelling ho to milta-julta naam (Close Match) dhoondhna
    if matched.empty:
        close_matches = get_close_matches(user_input, all_titles_lower, n=1, cutoff=0.4)
        if close_matches:
            closest_title = close_matches[0]
            matched = df[df['title_lower'] == closest_title]

    # Agar bilkul bhi koi match na mile
    if matched.empty:
        return []

    idx = matched.index[0]

    # Similarity scores calculate karke sort karna
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Top 10 recommendations (index 1 se 11 tak)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]

    return df['title'].iloc[movie_indices].tolist()