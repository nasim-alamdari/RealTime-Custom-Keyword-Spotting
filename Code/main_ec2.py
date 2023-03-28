# uvicorn main_ec2:app --reload --workers 1 --host 0.0.0.0 --port 8000
import io
from fastapi import FastAPI, File, Form
from fastapi.responses import JSONResponse
import os
import re
import numpy as np
from pathlib import Path
from scipy.io.wavfile import write, read
from model_realtime import preproc, train, predict, report_result, infer_rltime
from func import input_data


BASE_DIR = Path(__file__).resolve(strict=True).parent
app = FastAPI()


def proc_train(audio_data):
    
    keyword_dir = os.path.join(BASE_DIR , './content/target_kw/uploaded/')
    for file in Path(keyword_dir).glob("**/*.wav"):
        filename = os.path.basename(file).split('/')[-1]
        if '_predict_audio' not in filename :
            m = re.search('(.+?).wav', filename)
            KEYWORD = m.group(1)
            print("keyword is:", KEYWORD)
            
            # Load audio data using librosa
            fs, audio = read(file)

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
def eval_stream (KEYWORD, model,frame):
    model_settings = input_data.standard_microspeech_model_settings(label_count=3)

    test_spectrograms = input_data.to_micro_spectrogram(model_settings, frame)
    test_spectrograms = test_spectrograms[np.newaxis, :, :, np.newaxis]

    # fetch softmax predictions from the finetuned model:
    pred = model.predict(test_spectrograms)
    categorical_pred = np.argmax(pred, axis=1)
    return pred,categorical_pred


def predict_per_chunk(audio_data):
    
    keyword_dir = os.path.join(BASE_DIR , './content/target_kw/uploaded/')
    for file in Path(keyword_dir).glob("**/*.wav"):
        filename = os.path.basename(file).split('/')[-1]
        if '_predict_audio' in filename :
            m = re.search('(.+?)_predict_audio', filename)
            KEYWORD = m.group(1)
            print("keyword is:", KEYWORD)
            
            # Load audio data using librosa
            fs, audio = read(file)
            audio = audio / np.max(np.abs(audio))

    """# split audio to 1-sec chunks
    splits = []
    noSections = int(np.ceil(len(audio) / fs))
    for i in range(noSections):
        # get 1 second
        temp = audio[i*fs : i*fs + fs] # this is for mono audio
        # add to list
        splits.append(temp)"""
    window_size = int(0.025*fs)
    hop_length = int(window_size/2)

    # Apply a Hamming window to the audio data
    window = np.hamming(window_size)
    audio = audio[:len(audio)//window_size*window_size].reshape(-1, window_size) * window

    frames = np.array([])
    chunks = []
    # Loop over the frames and extract the audio data
    for i in range(audio.shape[0]):
        frames = np.append(frames, audio [i,:]) #audio[start:end]
        if len(frames) == fs:
            chunks.append(frames)
            frames = []

    # Iterate over frames and process them as needed
    model, ths = infer_rltime(KEYWORD) # TODO: this should load only once
    preds = []
    #for i in range(noSections):
    for i in range(len(chunks)):
        # Predict each 1-sec audio chunk
        audio_chunk = chunks[i] #np.frombuffer(frame, dtype=np.int16)
        pred, categorical_pred = eval_stream (KEYWORD, model,audio_chunk)
        if categorical_pred == 1:
            if pred[0][categorical_pred] >= 0.9:
                print(("Other Words"))
                preds.append("Other Words")
        elif categorical_pred == 2:
            if pred[0][categorical_pred] >= 0.9:
                print("KEYWORD")
                preds.append( "KEYWORD")
        elif categorical_pred == 0:
            if pred[0][categorical_pred] >= 0.8:
                print("Background Noise/Silence")
                preds.append( "Background Noise/Silence")
        
    return preds


@app.post("/train")
def make_prediction(file: bytes = File(...)):
    # Make prediction using audio file data
    result = proc_train(file)
    # Return prediction result as JSON
    return JSONResponse(content={"result": result})


@app.post("/predict")
def make_prediction(file: bytes = File(...)):
    # Make prediction using audio file data
    predictions = predict_per_chunk(file)
    # Return prediction result as JSON
    return JSONResponse(content={"prediction": predictions})



if __name__ == "__main__":
    uvicorn.run(main_ec2, host="0.0.0.0", port=8000)
