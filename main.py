from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from model import recommend_movies

app = FastAPI(title="Movie Recommender System")

# Templates setup
templates = Jinja2Templates(directory=".")

# Request Body structure
class MovieRequest(BaseModel):
    movie_name: str

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    """Frontend HTML page render karne ke liye"""
    return templates.TemplateResponse(request=request, name="index.html")

@app.post("/api/recommend")
def get_recommendation_api(data: MovieRequest):
    """JavaScript API endpoint jise frontend call karega"""
    results = recommend_movies(data.movie_name)
    
    if not results:
        return {"success": False, "message": f"Sorry! '{data.movie_name}' dataset me nahi mili."}
    
    return {"success": True, "recommendations": results}