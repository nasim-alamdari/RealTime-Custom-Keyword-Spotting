# run this command: streamlit run streamlit_ec2.py
import streamlit as st
import requests
import pyaudio
import os
from pathlib import Path
from model_realtime import record, preproc, train, predict, report_result, infer
#import websockets
#import asyncio
#import base64
#import json
import librosa
BASE_DIR = Path(__file__).resolve(strict=True).parent



# Web user interface
st.title('🎙️ Multilingual Custom Few-Shot Keyword Spotting App')

with st.expander('About this App'):
    st.markdown('''
    This Streamlit app performs demo of training and streaming evaluation of custom keyword spotting through streamlit and fastapi model serving.

    Steps:
    - `1. Type a custom keyword without any space` - e.g. "heyGPT"
    - `2. Upload you audio file for fine-tuning model.` - For security purposes, audio file will be deleted immediately after fine-tuning the model.
    - `3. Upload you audio file for prediction` - Prediction has 3 classes: (1)keyword, (2)otherword, or (3)background noise/silence

    ''')

# Step 1: Define custom keyword
st.markdown('##### :violet[Step 1: Enter Your Custom Keyword]')
KEYWORD = st.text_input('e.g. heyGPT')
st.write('The current Custom Keyword is: ', KEYWORD)

def send_audio_file(file, flag):
    # Define FastAPI endpoint URL
    if flag == "train":
        api_url = "http://0.0.0.0:8000/train"
        files = {"file": (KEYWORD +"_audio_train.wav", file.read(), "audio/wav")}
        st.write('Training may take few minutes...⌛')
    elif flag == "predict":
        api_url = "http://0.0.0.0:8000/predict"
        files = {"file": (KEYWORD +"_audio_predict.wav", file.read(), "audio/wav")}
        st.write('Prediction may take a minute...⌛')
        
    # Send POST request to FastAPI app
    response = requests.post(api_url, files=files)
    # Check if request was successful
    if response.status_code == 200:
        # Get prediction result from response
        result = response.json()
        # Display prediction result in Streamlit app
        if flag == "predict":
            st.write("Prediction result:", result)
        return result
    else:
        st.write("Error making result request")
        return None



def upload_train_file():
    keyword_dir = os.path.join(BASE_DIR , './content/target_kw/uploaded/')
    if not os.path.exists(keyword_dir):
        os.mkdir(keyword_dir)
        
    uploaded_wav = st.file_uploader("Upload your (.wav) audio file", type=["wav"], key="audio_file1")

    if uploaded_wav is not None:
        st.markdown('###### :green[Upload completed! 🏁]')     
        # Display the uploaded file
        st.write(uploaded_wav)
        
        # Read WAV file and Display audio waveform
        y, sr = librosa.load(uploaded_wav)
        st.audio(y, format='audio/wav', sample_rate=sr)

        uploaded_wav.name = KEYWORD + "_train_audio.wav" # rename audio before sending              
        file_path = os.path.join(keyword_dir, uploaded_wav.name)
        with open(file_path, "wb") as f: 
            st.success("File saved successfully!")

        # Create a button that the user must click after uploading the file
        st.markdown('##### :violet[Step 3: Fine-Tuning the KWS Model]')
        if st.button("Process Audio for Fine-Tuning"):
            # Do something with the uploaded file
            result = send_audio_file(uploaded_wav, "train")
            st.write("Train File processed!")
    else:
        st.write("file is not uploaded yet.")
        

def upload_predict_file():
    keyword_dir = os.path.join(BASE_DIR , './content/target_kw/uploaded/')
    if not os.path.exists(keyword_dir):
        os.mkdir(keyword_dir)
        
    uploaded_wav = st.file_uploader("Upload your (.wav) audio file", type=["wav"], key="audio_file2")

    if uploaded_wav is not None:
        st.markdown('###### :green[Upload completed! 🏁]')     
        # Display the uploaded file
        st.write(uploaded_wav)
        
        # Read WAV file and Display audio waveform
        y, sr = librosa.load(uploaded_wav)
        st.audio(y, format='audio/wav', sample_rate=sr)

        uploaded_wav.name = KEYWORD + "_predict_audio.wav" # rename audio before sending           
        file_path = os.path.join(keyword_dir, uploaded_wav.name)
        with open(file_path, "wb") as f: 
            st.success("File saved successfully!")

        # Create a button that the user must click after uploading the file
        st.markdown('##### :violet[Step 5: Prediction]')
        if st.button("Process Audio for Prediction"):
            # Do something with the uploaded file
            result = send_audio_file(uploaded_wav, "predict")
            st.write("Predict File processed!")
    else:
        st.write("file is not uploaded yet.")

        
st.markdown('##### :violet[Step 2: Uploading a file for training]')
upload_train_file()


st.markdown('##### :violet[Step 4: Uploading a file for prediction]')       
upload_predict_file()













