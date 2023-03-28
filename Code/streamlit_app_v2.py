# run this command: streamlit run streamlit_app.py
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

# def start_recording(KEYWORD, duration):
#     st.write("Please Repeat Your Custom Keyword ...")
#     fs, keyword_dir = record(KEYWORD,duration)

#     # segment the recorded audio
#     spk_segments_path = preproc (KEYWORD,keyword_dir,fs)
#     st.markdown('###### :green[Recording completed! 🏁]')
#     return spk_segments_path



# Step 1: Define custom keyword
st.markdown('###### :violet[Step 1: Enter Your Custom Keyword]')
KEYWORD = st.text_input('e.g. heyGPT')
st.write('The current Custom Keyword is: ', KEYWORD)


def send_audio_file(file):
    # Define FastAPI endpoint URL
    api_url = "http://localhost:8000/train"
    # Prepare audio file as multipart/form-data
    files = {"file": ("audio.wav", file.read(), "audio/wav")}
    # Send POST request to FastAPI app
    response = requests.post(api_url, files=files)
    # Check if request was successful
    if response.status_code == 200:
        # Get prediction result from response
        result = response.json()
        # Display prediction result in Streamlit app
        st.write("Prediction result:", result)
        return result
    else:
        st.write("Error making prediction request")
        return None

def start_uploading(KEYWORD):
    st.write("Please Upload Your Custom Keyword ...")
    keyword_dir = os.path.join(BASE_DIR , './content/target_kw/uploaded/',KEYWORD )
    if not os.path.exists(keyword_dir):
        os.mkdir(keyword_dir)
    uploaded_wav = st.file_uploader("Upload your (.wav) audio file", type=["wav"], upload_folder=keyword_dir)
    return uploaded_wav

def start_training(spk_segments_path):
    test_samples= train(KEYWORD, spk_segments_path)
    target_pred, nontarget_pred = predict(KEYWORD, test_samples)
    frr_val,far_val = report_result (target_pred, nontarget_pred)
    st.markdown('###### :green[Training completed! 🏁]')
    st.write("FRR and FAA are:", frr_val, far_val)
    train_done = True
    return train_done

def start_inference():
    st.write('Loading the fine-tuned model may take a minute...⌛')
    st.write('Please start speaking, result will be shown at the end...')
    result = infer(KEYWORD,duration=30)
    st.write("Inference Ended")
    return result


# Web user interface
st.title('🎙️ Real-Time Multilingual Custom Few-Shot Keyword Spotting App')

with st.expander('About this App'):
    st.markdown('''
    This Streamlit app performs training and real-time evaluation of custom keyword spotting.

    Steps:
    - `1. Type a custom keyword without any space` - e.g. "heyGPT"
    - `2. Record your voice for 30 seconds.` - For security purposes, audio will be deleted immediately after fine-tuning the model.
    - `3. Click on Train` - Fine tune the KWS through 5-shot learning
    - `4. Try the custom KWS in real-time` - 
    ''')


info_dict = {}
# Step 1: Define custom keyword
st.markdown('###### :violet[Step 1: Enter Your Custom Keyword]')
KEYWORD = st.text_input('e.g. heyGPT')
st.write('The current Custom Keyword is: ', KEYWORD)
info_dict["keyword"] = KEYWORD


#col1,col2,col3 = st.columns(3)
#col1.button('Record 🎙️')
#col2.button('Train')
#col3.button('Start Inference')
#col4.button('Stop Inference', on_click=stop_listening)

st.markdown('### :white[Upload Audio]')
duration = st.slider("Select recording duration (in seconds)", 15, 60, 20) # minimum of 20 second recording
st.markdown(':violet[Press "*Uploading*" and Repeat Your Custom Keyword.]')
if st.button('Record 🎙️'):
    uploaded_wav = start_uploading(KEYWORD)
    # Check if file was uploaded
    if uploaded_wav is not None:
        # Read WAV file
        y, sr = librosa.load(uploaded_wav)
        # Display audio waveform
        st.audio(y, format='audio/wav', sample_rate=sr)
        st.markdown('###### :green[Upload completed! 🏁]')
        # start training if file was uploaded
        result = send_audio_file(uploaded_wav)
    
# st.markdown('### :white[Train]')
# st.markdown('violet[Press "*Train*" to Fine-Tune the Model]')
# st.write('Training may take few minutes...⌛')
# if st.button('Train 🛠️'):
#     spk_segments_path = os.path.join(BASE_DIR , './content/target_kw/recording/',KEYWORD)
#     spk_segments_path = os.path.join(spk_segments_path, 'extractions')
#     train_done = start_training(spk_segments_path)
#     info_dict["train_done"] = train_done
    
# st.markdown('### :white[Real-Time Evaluation]')
# st.markdown(':violet[Press "*Start Inference*" to Test the Fine-Tuned Custom Keyword Spotting Model in Real-Time]')
# st.write('Result will be shown after 30 seconds...⌛')
# if st.button('Start Inference ▶️'):
#     #if info_dict["train_done"]:
#     result = start_inference()
#     info_dict["result"] = result
#     st.write("Results:", result)
