# run this command: streamlit run streamlit_app.py
import streamlit as st
import pyaudio
import os
from pathlib import Path
from model_realtime import record, preproc, train, predict, report_result, infer
#import websockets
#import asyncio
#import base64
#import json


# Session state
if 'text' not in st.session_state:
    st.session_state['text'] = 'Listening...'
    st.session_state['run'] = False

# Audio parameters 
st.sidebar.header('Audio Parameters')

FRAMES_PER_BUFFER = int(st.sidebar.text_input('Frames per buffer', 200))
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = int(st.sidebar.text_input('Rate', 16000))
p = pyaudio.PyAudio()


    
BASE_DIR = Path(__file__).resolve(strict=True).parent

def start_recording(KEYWORD):
    st.write("Please Repeat Your Custom Keyword ...")
    fs, keyword_dir = record(KEYWORD)

    # segment the recorded audio
    spk_segments_path = preproc (KEYWORD,keyword_dir,fs)
    st.markdown('###### :green[Recording completed! 🏁]')
    return spk_segments_path
    

def start_training(spk_segments_path):
    test_samples= train(KEYWORD, spk_segments_path)
    target_pred, nontarget_pred = predict(KEYWORD, test_samples)
    frr_val,far_val = report_result (target_pred, nontarget_pred)
    st.markdown('###### :green[Training completed! 🏁]')
    st.write("FRR and FAA are:", frr_val, far_val)
    train_done = True
    return train_done

def start_inference():
    st.write('Loading the fine-tuned model...')
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

st.markdown('###### :violet[Step 2: Press "*Record*" and Repeat Your Custom Keyword for 30 Seconds.]')
if st.button('Record 🎙️'):
    spk_segments_path = start_recording(info_dict["keyword"])
    info_dict["spk_segments_path"] = spk_segments_path
    
st.markdown('###### :violet[Step 3: Press "*Train*" to Fine-Tune the Model]')
st.write('Training may take few minutes...⌛')
if st.button('Train 🛠️'):
    spk_segments_path = os.path.join(BASE_DIR , './content/target_kw/recording/',KEYWORD)
    spk_segments_path = os.path.join(spk_segments_path, 'extractions')
    train_done = start_training(spk_segments_path)
    info_dict["train_done"] = train_done
    
st.markdown('###### :violet[Step 4: Press "*Start Inference*" to Test the Fine-Tuned Custom Keyword Spotting Model in Real-Time]')
if st.button('Start Inference ▶️'):
    #if info_dict["train_done"]:
    result = start_inference()
    info_dict["result"] = result
    st.write("Results:", result)
