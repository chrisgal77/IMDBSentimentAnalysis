from typing import Optional
import pathlib
from fastapi import FastAPI


app = FastAPI()

BASE_DIR = pathlib.Path(__file__).resolve().parent


@app.on_event("startup")
def on_startup():
    # load model
    # global MODEl
    # MODEL = load_model(MODEL_PATH)
    pass


@app.get('/') # /?q=cos
def read_index(q: Optional[str] = None):
    query = q or 'hello'
    return {'query' : query}


