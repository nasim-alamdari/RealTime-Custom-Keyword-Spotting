""" To see results do the following steps:
 - in terminal run: uvicorn main:app --reload --workers 1 --host 0.0.0.0 --port 8000
 - Navigate to following link in your browser: http://localhost:8000/ping
 
 
Below is the explanation of the command:
--reload: enables auto-reload so the server will restart after changes are made to the code base.
--workers 1: provides a single worker process.
--host 0.0.0.0 :defines the address to host the server on.
--port 8000: defines the port to host the server on.
--main:app :tells uvicorn where it can find the FastAPI ASGI application. In this case, within the main.py file, you will find the ASGI app app = FastAPI().

"""
from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel
from model import predict, convert

app = FastAPI()

# pydantic models
class StockIn(BaseModel):
    ticker: str
    days: int

class StockOut(StockIn):
    forecast: dict

@app.post("/predict", response_model=StockOut, status_code=200)
def get_prediction(payload: StockIn):
    ticker = payload.ticker
    days = payload.days

    prediction_list = predict(ticker, days)

    if not prediction_list:
        raise HTTPException(status_code=400, detail="Model not found.")

    response_object = {
        "ticker": ticker, 
        "days": days,
        "forecast": convert(prediction_list)}
    return response_object

"""@app.get("/ping")
def pong():
    return {"ping": "pong!"}"""