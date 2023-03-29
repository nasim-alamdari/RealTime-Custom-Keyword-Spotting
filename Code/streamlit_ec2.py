# run this command: streamlit run streamlit_ec2.py
import streamlit as st
import requests
import os
from pathlib import Path
from model_realtime import record, preproc, train, predict, report_result, infer
#import websockets
#import asyncio
#import base64
#import json
import librosa
from scipy.io.wavfile import write, read
import subprocess
import shutil
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
st.markdown('### :white[Enter Custom Keyword]')
st.markdown(':violet[Step 1: Please Enter Your Custom Keyword with No Spcae Between Words (maximum 2 words)]')
KEYWORD = st.text_input('e.g. heyGPT')
st.write('The current Custom Keyword is: ', KEYWORD)


def del_prevFiles (folder_path):
    # get a list of all files in the folder
    file_list = os.listdir(folder_path)

    # loop through the files and delete them
    for filename in file_list:
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
        except Exception as e:
            print(e)


def send_audio_file(file, flag):
    # Define FastAPI endpoint URL
    if flag == "train":
        api_url = "http://0.0.0.0:8000/train"
        files = {"file": (KEYWORD +"_audio_train.wav", file.read(), "audio/wav")}
        st.write('Training may take 2-3 minutes...⌛')
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
        
    # delete any existing audio files before any uploading
    del_prevFiles (keyword_dir) 
    
    uploaded_wav = st.file_uploader("Upload your (.wav) audio file that has **at least 10 times keyword (only) repeated**", type=["wav"], key="audio_file1")
    

    if uploaded_wav is not None:
        st.markdown('###### :green[Upload completed! 🏁]')     
        # Display the uploaded file
        st.write(uploaded_wav)
        
        # Read WAV file and Display audio waveform
        sr, y = read(uploaded_wav)
        if len(y)/sr < 15:
            st.error("Please upload an audio file that is longer than 15 seconds, and has at least 10 times keyword (only) repeated", icon="🚨")
        st.audio(y, format='audio/wav', sample_rate=sr)

        uploaded_wav.name = KEYWORD +".wav" # rename audio before sending              
        file_path = os.path.join(keyword_dir, uploaded_wav.name)
        with open(file_path, "wb") as f: 
            write(file_path,  sr, y)
            subprocess.run('ffmpeg -hide_banner -loglevel error -y -i file_path -acodec pcm_s16le -ac 1 -ar 16000 file_path', shell=True)
            #st.success("File saved successfully!")

        # Create a button that the user must click after uploading the file
        st.markdown('##### :violet[Step 3: Fine-Tuning the KWS Model]')
        if st.button("Process Audio for Fine-Tuning"):
            result = send_audio_file(uploaded_wav, "train")
            st.success('###### :green[Training completed! 🏁]')
            
            # delete existing audio files after training
            #os.remove(file_path)
            shutil.rmtree(os.path.join(keyword_dir, 'extractions'))
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
        sr, y = read(uploaded_wav)
        st.audio(y, format='audio/wav', sample_rate=sr)

        uploaded_wav.name = KEYWORD + "_predict_audio.wav" # rename audio before sending           
        file_path = os.path.join(keyword_dir, uploaded_wav.name)
        with open(file_path, "wb") as f: 
            write(file_path,  sr, y)
            subprocess.run('ffmpeg -hide_banner -loglevel error -y -i file_path -acodec pcm_s16le -ac 1 -ar 16000 file_path', shell=True)
            #st.success("File saved successfully!")

        # Create a button that the user must click after uploading the file
        st.markdown('##### :violet[Step 5: Prediction]')
        if st.button("Process Audio for Prediction"):
            result = send_audio_file(uploaded_wav, "predict")
            st.success('###### :green[Prediction completed! 🏁]')
            # delete existing audio files after prediction
            os.remove(file_path)
    else:
        st.write("file is not uploaded yet.")

st.markdown('### :white[Train]')   
st.markdown('##### :violet[Step 2: Uploading an audio file for training]')
upload_train_file()

st.markdown('### :white[Prediction]')   
st.markdown('##### :violet[Step 4: Uploading an audio file for prediction]')       
upload_predict_file()













