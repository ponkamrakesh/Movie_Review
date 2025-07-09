from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import os

app = FastAPI()

model_dir = "model"
model_path = os.path.join(model_dir, "sentiment_gru_model.h5")

model = load_model(model_path)

with open(os.path.join(model_dir, "tokenizer.pkl"), "rb") as f:
    tokenizer = pickle.load(f)
max_len = 50  # same as used during training

templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "result": None})

@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, text: str = Form(...)):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=max_len, padding='post')
    prob = model.predict(padded)[0][0]
    label = "Positive" if prob >= 0.5 else "Negative"
    confidence = f"{prob * 100:.2f}%"
    result = f"{label} (Confidence: {confidence})"
    return templates.TemplateResponse("index.html", {"request": request, "result": result})
