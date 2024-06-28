from fastapi import FastAPI, Depends
import redis
import json
from typing import List, Tuple
from pydantic import BaseModel

app = FastAPI()

# Configure Redis connection
redis_client = redis.Redis(host='redis', port=6379, db=0)

class Prediction(BaseModel):
    station_name: str
    bikes_available: float

def get_predictions_from_cache() -> List[Tuple[str, float]]:
    cached_predictions = redis_client.get('predictions')
    if cached_predictions:
        return json.loads(cached_predictions)
    return None

def set_predictions_to_cache(predictions: List[Tuple[str, float]]):
    redis_client.set('predictions', json.dumps(predictions))

def load_predictions_from_file() -> List[Tuple[str, float]]:
    with open('/app/data/predictions.json', 'r') as f:
        predictions = json.load(f)
    return predictions

@app.get("/")
def read_root():
    return {"message": "Welcome to the Velib Predictions API. Visit /predictions to get the data."}

@app.get("/predictions", response_model=List[Prediction])
def get_predictions():
    predictions = get_predictions_from_cache()
    if not predictions:
        predictions = load_predictions_from_file()
        set_predictions_to_cache(predictions)
    return [{"station_name": name, "bikes_available": bikes} for name, bikes in predictions]
