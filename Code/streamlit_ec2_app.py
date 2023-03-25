# streamlit run streamlit_ec2_app.py
import asyncio
import websockets
import streamlit as st
import numpy as np
import pyaudio
import time
import io
import resampy

# Session state
if 'text' not in st.session_state:
    st.session_state['text'] = 'Listening...'
    st.session_state['run'] = False
    st.session_state['load_model'] = 0
    

def on_message(ws, message):
    print("Received message: ", message)

def on_error(ws, error):
    print("Error occurred: ", error)

def on_close(ws):
    print("Connection closed")

def on_open(ws):
    print("Connection opened")


# Audio parameters 
st.sidebar.header('Audio Parameters')
FRAMES_PER_BUFFER = int(st.sidebar.text_input('Frames per buffer', 200))
RATE = int(st.sidebar.text_input('Rate', 16000))
TH = int(st.sidebar.text_input('Detection Confidence (%)', 90))

# Initialize PyAudio
p = pyaudio.PyAudio()

# Set up audio stream for recording
FORMAT = pyaudio.paInt16
CHANNELS = 1

#============== Web user interface ============================
st.title('🎙️ Real-Time Multilingual Custom Few-Shot Keyword Spotting App')


#image = Image.open('../imgs/streamlit_img.png')
#st.image(image, caption='KeywordSpotting')
with st.expander('About this App'):
    st.markdown('''
    To design, implement and deploy a lightweight Multilingual Custom Keyword spotting on an Edge Device.

    Steps:
    - `1. Type a custom keyword without any space between words` - e.g. "heyGPT"
    - `2 & 3. Click on Record and Train` - to Record your voice for 20-30 seconds. For security purposes, audio will be deleted immediately after fine-tuning the model. Fine tunining the KWS model will be done through 5-shot learning
    - `4. Click on Start Inference`: to try the custom KWS in real-time  
    ''')

#===========================================================
async def send_keyword (KEYWORD):
    st.write("Updating Keyword...")
    ws = websocket.WebSocketApp("ws://<your_ec2_instance_ip>:<your_websocket_port>/keyword",
                                on_open=on_open,
                                on_message=on_message,
                                on_error=on_error,
                                on_close=on_close)

    await ws.send(KEYWORD.totext())
    await ws.recv()
    ws.close() 


# Step 1: Define custom keyword
st.markdown('### :white[Enter Custom Keyword]')
st.markdown(':violet[Please Enter Your Custom Keyword with No Spcae Between Words (maximum 2 words)]')
KEYWORD = st.text_input('e.g. heyGPT')
button = st.button("Submit Keyword")
if button and KEYWORD != "":
    asyncio.run(send_keyword(KEYWORD))
    st.write('The Custom Keyword is Now: ', KEYWORD)
  

#===========================================================


async def record_audio(duration):
    """
    Record audio for a given duration (in seconds).
    Returns the recorded audio data as a bytes object.
    """
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=FRAMES_PER_BUFFER)
    frames = []
    for i in range(int(RATE / CHUNK * duration)):
        data = stream.read(CHUNK)
        frames.append(data)
    audio_frames = b"".join(frames)
    audio_frames = resampy.resample(audio_frames, RATE, 16000)
    stream.stop_stream()
    stream.close()
    p.terminate()
    st.write("Recording completed.")
    return audio_frames

# the "save_audio_file" will not be used.
async def save_audio_file(audio_data, filename):
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b"".join(audio_data))
        wf.close()
        

async def send_audio_train(audio_data):
    ws = websocket.WebSocketApp("ws://<your_ec2_instance_ip>:<your_websocket_port>/keyword",
                                on_open=on_open,
                                on_message=on_message,
                                on_error=on_error,
                                on_close=on_close)
    
    await ws.send(audio_data.totext())
    st.write("Recorded audio is sent to the EC2 instance.")
    await ws.recv()
    st.write("Training completed")
    ws.close()   

st.markdown('### :white[Record and Train]')
st.markdown(':violet[Press "*Record*" and Repeat Your Custom Keyword for 30 Seconds.]')
st.markdown(':violet[Audio Will be Sent to EC2 Instance to Fine-Tune the Model]')
duration = st.slider("Select recording duration (in seconds)", 20, 60, 25) # minimum of 20 second recording
if st.button('Record 🎙️ and Train 🛠️'):
    st.spinner(text="In progress...")
    audio_data = record_audio(duration)
    asyncio.run(send_audio_train(audio_data))    
    
#===========================================================

async def predict_audio():
    ws = websocket.WebSocketApp("ws://<your_ec2_instance_ip>:<your_websocket_port>/inference",
                                on_open=on_open,
                                on_message=on_message,
                                on_error=on_error,
                                on_close=on_close)
    
    while st.session_state['run']:
        # capture real-time audio as bytes
        #data = sd.rec(int(48000 * 1), samplerate=48000, channels=1, dtype='int16')
        stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True,frames_per_buffer=CHUNK)
        frames = []
        for i in range(int(RATE / CHUNK * duration)):
            data = stream.read(CHUNK)
            frames.append(data)
            if len(frames) == (1*RATE):
                frames = resampy.resample(frames, RATE, 16000)
                await ws.send(frames.tobytes())
                # wait for prediction result
                result = await ws.recv()
                # display prediction result
                st.write(result)
                frames = []
    stream.stop_stream()
    stream.close()
    p.terminate()   

    
st.markdown('### :white[Real-Time Evaluation]')
st.markdown(':violet[Press "*Start Inference*" to Test the Fine-Tuned Custom Keyword Spotting Model in Real-Time]')
if st.button('Stop Inference ⏹️'):
    st.session_state['run'] = False
    
if st.button('Start Inference ▶️'):
    st.session_state['run'] = True
    asyncio.run(predict_audio())
    


#option = st.sidebar.selectbox("Select an option", ["Train", "Predict"])

    


