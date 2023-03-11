# Nasim Alamdari and Christos Magganas
# last Update March 2023
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
from model import train,predict, report_result
from typing import List, Optional
from pathlib import Path

app = FastAPI()

# pydantic models
class AudioIn(BaseModel):
    keyword: Optional[str] = None
    keyword_dir: Optional[str] = None

class AudioOut(AudioIn):
    FRR_and_FAR: List[float]


@app.post("/predict", response_model=AudioOut, status_code=200)
def get_prediction(payload: AudioIn):
    keyword = payload.keyword
    keyword_dir = payload.keyword_dir
    
    keyword, test_samples= train(keyword, keyword_dir)
    target_pred, nontarget_pred = predict(keyword, test_samples)
    frr_val,far_val = report_result (target_pred, nontarget_pred)
    

    if (not frr_val) or (not far_val):
        raise HTTPException(status_code=400, detail="Model not found.")

    response_object = {
        "keyword": keyword, 
        "FRR_and_FAR": [frr_val,far_val]}
    return response_object

"""@app.get("/ping")
def pong():
    return {"ping": "pong!"}"""
