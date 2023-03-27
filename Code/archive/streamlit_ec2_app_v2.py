# streamlit run streamlit_ec2_app.py
import asyncio
import websockets
import streamlit as st
import numpy as np
#import pyaudio
import time
import io
import resampy
import webrecorder

# Session state
if 'text' not in st.session_state:
    st.session_state['text'] = 'Listening...'
    st.session_state['run'] = False
    st.session_state['load_model'] = 0



# Audio parameters 
st.sidebar.header('Audio Parameters')
FRAMES_PER_BUFFER = int(st.sidebar.text_input('Frames per buffer', 200))
RATE = int(st.sidebar.text_input('Rate', 16000))
TH = int(st.sidebar.text_input('Detection Confidence (%)', 90))

# create an instance of the Recorder class
recorder = webrecorder.Recorder()

# Set up audio stream for recording
#FORMAT = pyaudio.paInt16
CHANNELS = 1
CHUNK = FRAMES_PER_BUFFER


# get the index of the microphone in the list of available audio devices

# get available input devices and their names
devices = sd.query_devices()
input_devices = [(i, d['name']) for i, d in enumerate(devices) if d['max_input_channels'] > 0]

# create dropdown menu for selecting input device
default_device_index = p.get_default_input_device_info()['index']
selected_device_index = st.selectbox('Select microphone device', input_devices, index=default_device_index)


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
    async with websockets.connect("ws://35.87.244.144:8000/keyword") as ws:

        await ws.send(KEYWORD)
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
    frames = recorder.start(format='int16', channels= CHANNELS, sample_rate=RATE, chunk_size=FRAMES_PER_BUFFER, duration=duration)
    """stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=FRAMES_PER_BUFFER,
                    input_device_index=selected_device_index)
    frames = []
    for i in range(int(RATE / CHUNK * duration)):
        data = stream.read(CHUNK)
        frames.append(data)"""
    audio_frames = b"".join(frames)
    #audio_frames = resampy.resample(audio_frames, RATE, 16000)
    #stream.stop_stream()
    #stream.close()
    #p.terminate()
    recording = recorder.stop()
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
    async with websockets.connect("ws://35.87.244.144:8000/train") as ws:
    
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
    async with websockets.connect("ws://35.87.244.144:8000/inference") as ws:
        while st.session_state['run']:
            # capture real-time audio as bytes
            #stream = p.open(format=FORMAT,
            #                channels=CHANNELS,
            #                rate=RATE, input=True,
            #                frames_per_buffer=CHUNK,
            #                input_device_index=selected_device_index,)
            #frames = []
            #for i in range(int(RATE / CHUNK * duration)):
            #    data = stream.read(CHUNK)
            #    frames.append(data)
            #    if len(frames) == (1*RATE):
            frames = recorder.start(format='int16', channels= CHANNELS, sample_rate=RATE, chunk_size=FRAMES_PER_BUFFER, duration=duration)
            #frames = resampy.resample(frames, RATE, 16000)
            await ws.send(frames.tobytes())
            # wait for prediction result
            result = await ws.recv()
            # display prediction result
            st.write(result)
                    #frames = []
        #stream.stop_stream()
        #stream.close()
        #p.terminate()
        recording = recorder.stop()
        ws.close()   

    
st.markdown('### :white[Real-Time Evaluation]')
st.markdown(':violet[Press "*Start Inference*" to Test the Fine-Tuned Custom Keyword Spotting Model in Real-Time]')
if st.button('Stop Inference ⏹️'):
    st.session_state['run'] = False
    
if st.button('Start Inference ▶️'):
    st.session_state['run'] = True
    asyncio.run(predict_audio())
    


#option = st.sidebar.selectbox("Select an option", ["Train", "Predict"])

    


