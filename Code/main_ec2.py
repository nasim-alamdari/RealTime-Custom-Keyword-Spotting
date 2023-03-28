# uvicorn main_ec2:app --reload --workers 1 --host 0.0.0.0 --port 8000
import io
import librosa
import numpy as np
from fastapi import FastAPI, File, Form
from fastapi.responses import JSONResponse
import os
from pathlib import Path
from model_realtime import preproc, train, predict, report_result, infer_rltime
import re

BASE_DIR = Path(__file__).resolve(strict=True).parent
app = FastAPI()


def proc_train(audio_data):
    
    keyword_dir = os.path.join(BASE_DIR , './content/target_kw/uploaded/')
    for file in Path(keyword_dir).glob("**/*.wav"):
        filename = os.path.splitext(file)[0]
        if '_train_audio' in filename :
            m = re.search('(.+?)_train_audio', filename)
            KEYWORD = m.group(1)
            print("keyword is:", KEYWORD)
            
            # Load audio data using librosa
            #audio, fs = librosa.load(io.BytesIO(audio_data), sr=None, mono=True)
            audio, fs = librosa.load(file, sr=None, mono=True)

    # segment the recorded audio
    spk_segments_path = preproc (KEYWORD,keyword_dir,fs)
    print('segmentation completed!')

    test_samples= train(KEYWORD, spk_segments_path)
    target_pred, nontarget_pred = predict(KEYWORD, test_samples)
    frr_val,far_val = report_result (target_pred, nontarget_pred)
    print('Training completed!')
    print("FRR and FAA are:", frr_val, far_val)
    return [frr_val, far_val]
    


# Define prediction function
def eval_stream (KEYWORD, model,frames):
    model_settings = input_data.standard_microspeech_model_settings(label_count=3)

    test_spectrograms = input_data.to_micro_spectrogram(model_settings, frames)
    test_spectrograms = test_spectrograms[np.newaxis, :, :, np.newaxis]

    # fetch softmax predictions from the finetuned model:
    pred = model.predict(test_spectrograms)
    categorical_pred = np.argmax(pred, axis=1)
    return pred,categorical_pred


def predict_per_chunk(audio_data):
    
    keyword_dir = os.path.join(BASE_DIR , './content/target_kw/uploaded/')
    for file in Path(keyword_dir).glob("**/*.wav"):
        filename = os.path.splitext(file)[0]
        if '_predict_audio' in filename :
            m = re.search('(.+?)_train_audio', filename)
            KEYWORD = m.group(1)
            print("keyword is:", KEYWORD)
            
            # Load audio data using librosa
            #audio, fs = librosa.load(io.BytesIO(audio_data), sr=None, mono=True)
            audio, fs = librosa.load(file, sr=None, mono=True)

    # Define frame size and overlap
    frame_size = fs # equivalend to 1-sec audio chunk
    overlap = 0.5 # 50% overlap

    # Calculate hop length based on overlap
    hop_length = int(frame_size * overlap)

    # Generate frames with overlap using a sliding Hann window approach to generate overlapping frames
    #frames = librosa.util.frame(audio, frame_length=frame_size, hop_length=hop_length)
    window = librosa.filters.get_window('hann', frame_size, fftbins=True)
    frames = librosa.util.windows.sliding_window_view(audio, window.shape[0], hop_length=hop_length)

    # Iterate over frames and process them as needed
    preds = []
    for frame in frames:
        # Predict each 1-sec audio chunk
        audio_chunk = np.frombuffer(frame, dtype=np.int16)
        model, ths = infer_rltime(KEYWORD) # TODO: this should load only once
        pred, categorical_pred = eval_stream (KEYWORD, model,audio_chunk)
        if categorical_pred == 1:
            if pred[0][categorical_pred] >= ths[0]:
                preds.append("Other Words")
        elif categorical_pred == 2:
            if pred[0][categorical_pred] >= (ths[1]):
                preds.append( "KEYWORD")
        elif categorical_pred == 0:
            if pred[0][categorical_pred] >= ths[2]:
                preds.append( "Background Noise/Silence")
        
    return preds.tolist()


@app.post("/train")
async def make_prediction(file: bytes = File(...)):
    # Make prediction using audio file data
    print("file:",file)
    result = proc_train(file)
    # Return prediction result as JSON
    return JSONResponse(content={"result": result})


@app.post("/predict")
async def make_prediction(file: bytes = File(...)):
    # Make prediction using audio file data
    print("file:",file)
    predictions = predict_per_chunk(file)
    # Return prediction result as JSON
    return JSONResponse(content={"prediction": predictions})



if __name__ == "__main__":
    uvicorn.run(main_ec2, host="0.0.0.0", port=8000)
