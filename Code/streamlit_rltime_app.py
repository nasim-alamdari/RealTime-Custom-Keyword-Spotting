# run this command: streamlit run streamlit_rltime_app.py
import streamlit as st
import pyaudio
import os
from pathlib import Path
from model_realtime import record, preproc, train, predict, report_result, infer, infer_rltime

# for inference:
import numpy as np
import struct
import time
from func import input_data
import asyncio
import concurrent.futures
#import websockets

#import base64
#import json


# Session state
if 'text' not in st.session_state:
    st.session_state['text'] = 'Listening...'
    st.session_state['run'] = False
    st.session_state['load_model'] = 0
    

# Audio parameters 
st.sidebar.header('Audio Parameters')
FRAMES_PER_BUFFER = int(st.sidebar.text_input('Frames per buffer', 400))
RATE = int(st.sidebar.text_input('Rate', 16000))
TH = int(st.sidebar.text_input('Detection Confidence (%)', 90))
SESS = False
#============== Web user interface ============================
st.title('🎙️ Real-Time Multilingual Custom Keyword Spotting App')

with st.expander('About this App'):
    st.markdown('''
    This Streamlit app performs training and real-time evaluation of custom keyword spotting.

    Steps:
    - `1. Type a custom keyword without any space` - e.g. "heyGPT"
    - `2. Record your voice for 30 seconds.` - For security purposes, audio will be deleted immediately after fine-tuning the model.
    - `3. Click on Train` - Fine tune the KWS through 5-shot learning
    - `4. Try the custom KWS in real-time` -
    ''')



# Step 1: Define custom keyword
st.markdown('### :white[Enter Custom Keyword]')
st.markdown(':violet[Please Enter Your Custom Keyword with No Spcae Between Words (maximum 2 words)]')
KEYWORD = st.text_input('e.g. heyGPT')
button = st.button("Submit Keyword")
if button and KEYWORD != "":
    st.write('The Custom Keyword is Now: ', KEYWORD)
    
    
    
#===========================================================


BASE_DIR = Path(__file__).resolve(strict=True).parent

def start_recording(KEYWORD,duration):
    #st.write("Please Repeat Your Custom Keyword ...")
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        
        countdown_text = st.empty() # create an empty slot for the countdown text
        future = executor.submit(record, KEYWORD,duration)

        for i in range(duration, 0, -1):
            countdown_text.text(f"Recording will stop in {i} seconds...") # update the text in the slot
            time.sleep(1) # wait for one second before updating the text

        #countdown_text.text("Recording stopped.") 
        fs, keyword_dir = future.result()  

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



#========== Inference functions========================
def _find_input_device():
    device_index = None            
    for i in range( p.get_device_count() ):     
        devinfo = p.get_device_info_by_index(i)   
        print( "Device %d: %s"%(i,devinfo["name"]) )

        for record_name in ["mic","input"]:
            if record_name in devinfo["name"].lower():
                print( "Found an input: device %d - %s"%(i,devinfo["name"]) )
                device_index = i
                return device_index

    if device_index == None:
        print( "No preferred input found; using default input device." )

    return device_index


p = pyaudio.PyAudio()
CHANNELS = 1
RATE = 16000
INPUT_BLOCK_TIME = 0.0125
CHUNK = int(RATE*INPUT_BLOCK_TIME) # FRAMES_PER_BUFFER
device_index = _find_input_device()
stream = p.open(format=pyaudio.paFloat32,
                    channels=CHANNELS,
                    rate=RATE,
                    output=True,
                    input=True,
                    input_device_index = device_index,
                    frames_per_buffer=CHUNK)

    
def eval_stream (KEYWORD, model,frames):
    model_settings = input_data.standard_microspeech_model_settings(label_count=3)

    #===========================================
    # Test the trained FS-KWS model on test sets
    test_spectrograms = input_data.to_micro_spectrogram(model_settings, frames)
    #print(np.asarray(test_spectrograms).shape)
    test_spectrograms = test_spectrograms[np.newaxis, :, :, np.newaxis]
    #print(np.asarray(test_spectrograms).shape)

    # fetch softmax predictions from the finetuned model:
    pred = model.predict(test_spectrograms)
    categorical_pred = np.argmax(pred, axis=1)
    return pred,categorical_pred



def start_inference(SESS):
    if SESS and st.session_state['load_model']==0:
        st.write('Loading the fine-tuned model may take a few seconds...⌛')
        model, ths = infer_rltime(KEYWORD)
        st.write('Please start speaking...')
        st.session_state['load_model']=1
    frames = np.array([])
    result = []
    while SESS ==True:
    #while st.session_state['run']:
        #for i in range(int(duration*RATE/CHUNK)): #go for a LEN seconds
        block = np.fromstring(stream.read(CHUNK, exception_on_overflow = False),dtype=np.float32)
        frames = np.append(frames, block)
        
        if len(frames) == RATE: # 1-sec audio captures
            #t = time.time()
            pred, categorical_pred = eval_stream (KEYWORD, model,frames)
            
            if categorical_pred == 1:
                if pred[0][categorical_pred] >= ths[0]:
                    result.append("Other Words")
                    st.markdown(':orange[Other Words]')
            elif categorical_pred == 2:
                if pred[0][categorical_pred] >= (TH/100):
                    result.append("KEYWORD")
                    st.markdown(':orange[KEYWORD]')
            elif categorical_pred == 0:
                if pred[0][categorical_pred] >= ths[2]:
                    result.append("Background Noise/Silence")
                    st.markdown(':orange[Background Noise/Silence]')
            frames = []
            #st.write("processing time for a chunk:", time.time() - t)
    

    stream.stop_stream()
    stream.close()
    p.terminate()
    return st.write("Inference Ended")

    #return result
#========== End of Inference functions========================




#col1,col2,col3 = st.columns(3)
#col1.button('Record 🎙️')
#col2.button('Train')
#col3.button('Start Inference')
#col4.button('Stop Inference', on_click=stop_listening)

st.markdown('### :white[Record Audio]')
duration = st.slider("Select recording duration (in seconds)", 15, 60, 20) # minimum of 20 second recording
st.markdown(f':violet[Press "*Record*" and Repeat Your Custom Keyword for {duration} Seconds.]')
#st.markdown(':violet[Audio Will be Sent to Fine-Tune the Model]')
if st.button('Record 🎙️'):
    st.spinner(text="In progress...")
    spk_segments_path = start_recording(KEYWORD,duration)
    #info_dict["spk_segments_path"] = spk_segments_path
 
 
st.markdown('### :white[Train]')
st.markdown(':violet[Press "*Train*" to Fine-Tune the Model]')
st.write('Training may take 2-3 minutes...⌛')
if st.button('Train 🛠️'):
    st.spinner(text="In progress...")
    spk_segments_path = os.path.join(BASE_DIR , './content/target_kw/recording/',KEYWORD)
    spk_segments_path = os.path.join(spk_segments_path, 'extractions')
    train_done = start_training(spk_segments_path)
    

st.markdown('### :white[Real-Time Evaluation]')
st.markdown(':violet[Press "*Start Inference*" to Test the Fine-Tuned Custom Keyword Spotting Model in Real-Time]')
#col1,col2 = st.columns(2)  
if st.button('Stop ⏹️'):
    #del st.session_state['run']
    #st.session_state['run'] = False
    #st.session_state.update(st.session_state)
    SESS = False
    st.session_state['load_model']==1
    start_inference(False)
    
elif st.button('Start Inference ▶️'):
    #if info_dict["train_done"]:
    #st.session_state['run'] = True
    SESS = True
    start_inference(SESS)

    
    
    

