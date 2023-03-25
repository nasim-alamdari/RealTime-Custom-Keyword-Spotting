import uvicorn
from fastapi import FastAPI, WebSocket
from typing import List
import numpy as np
import os
from pathlib import Path
from scipy.io.wavfile import write, read
from model_realtime import record, preproc, train, predict, report_result, infer_rltime

# Define FastAPI app
app = FastAPI()

BASE_DIR = Path(__file__).resolve(strict=True).parent


# Define keyword function
@app.websocket("/keyword")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        # Receive new custom keyword
        KEYWORD = await websocket.receive_text()
        await websocket.send_text("New keyword recieved")
        

async def sgment_train(audio_data):
    # saving recorded audio as .wav file (we need it for speaker segmentation) 
    fs = 16000
    keyword_dir = os.path.join(BASE_DIR , './content/target_kw/recording/',KEYWORD )
    if not os.path.exists(keyword_dir):
        os.mkdir(keyword_dir)
    record_name = str(KEYWORD)+'.wav'
    write(os.path.join(keyword_dir,record_name),  fs, audio_data)
    
    # speaker segmentation
    spk_segments_path = preproc (KEYWORD,keyword_dir,fs)
    
    # training and offline evaluation:
    test_samples= train(KEYWORD, spk_segments_path)
    target_pred, nontarget_pred = predict(KEYWORD, test_samples)
    frr_val,far_val = report_result (target_pred, nontarget_pred)
    return "Training completed."


    
async def evaluate(audio_chunk):
    
    model, ths = infer_rltime(KEYWORD) # TODO: this should load only once
    
    pred, categorical_pred = eval_stream (KEYWORD, model,audio_chunk)
    if categorical_pred == 1:
        if pred[0][categorical_pred] >= ths[0]:
            return "Other Words"
    elif categorical_pred == 2:
        if pred[0][categorical_pred] >= (TH/100):
            return "KEYWORD"
    elif categorical_pred == 0:
        if pred[0][categorical_pred] >= ths[2]:
            return "Background Noise/Silence"
            
                
# Define training function
@app.websocket("/train")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        # Receive audio data from client
        audio_data = await websocket.receive_bytes()
        await sgment_train(audio_data)
        await websocket.send_text("Training completed...")
        
        
# Define inference function
@app.websocket("/inference")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        # Receive audio data from client
        audio_chunk = await websocket.receive_bytes()
        # This is a placeholder that simply returns the mean of the audio data
        audio_array = np.frombuffer(audio_data, dtype=np.int16)
        # Call inference function with audio data
        result = await evaluate(audio_chunk)
        # Send result to client
        #await websocket.send_bytes(result)
        await websocket.send_text(result)
        

if __name__ == "__main__":
    uvicorn.run(app_ec2, host="0.0.0.0", port=8000)

