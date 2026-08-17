# 🎬 Movie Recommendation System

A lightweight, content-based **Movie Recommendation System** built using **FastAPI**, **Scikit-Learn (TF-IDF & Cosine Similarity)**, and a modern single-page frontend (**HTML5, CSS3, JavaScript**).

This application accepts a movie name as input and recommends the **Top 10 most similar movies** based on genre and overview plot matching. It also features **fuzzy search/spelling tolerance** to handle misspelled user queries smoothly.

---

## ✨ Features

- **Content-Based Filtering:** Recommends movies by analyzing textual features (`genre` + `overview`) using TF-IDF vectorization.
- **Cosine Similarity Engine:** Calculates semantic closeness between vector embeddings to rank top matches.
- **Fuzzy Search & Spelling Tolerance:** Utilizes Python's `difflib.get_close_matches` to suggest accurate results even when movie titles are misspelled.
- **FastAPI Async Backend:** High-performance RESTful API endpoints with Jinja2 template integration.
- **All-in-One Frontend:** Clean, responsive dark-mode UI built with raw HTML, CSS, and asynchronous JavaScript (`fetch` API) for smooth SPA-like interactions without full page reloads.

---

## 🛠️ Tech Stack

| Domain | Technology |
| :--- | :--- |
| **Language** | Python 3.x |
| **Backend Framework** | FastAPI, Uvicorn |
| **Machine Learning / NLP** | Scikit-Learn (`TfidfVectorizer`, `cosine_similarity`), Pandas |
| **Frontend** | HTML5, CSS3, Vanilla JavaScript |
| **Templating Engine** | Jinja2 |

---

## 📁 Project Structure

```text
movie_project/
│
├── model.py           # Machine Learning logic (Data, TF-IDF, Cosine Similarity & Fuzzy Search)
├── main.py            # FastAPI backend application & API routes
├── templates/
│   └── index.html     # Single-page interface (HTML + Embedded CSS + Vanilla JS)
├── requirements.txt   # Python project dependencies
└── README.md          # Project documentation
