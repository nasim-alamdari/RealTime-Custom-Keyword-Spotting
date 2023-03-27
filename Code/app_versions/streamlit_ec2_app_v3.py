# streamlit run streamlit_ec2_app.py
import asyncio
import websockets
import streamlit as st
import numpy as np
import pyaudio
import time
import io
import resampy
import av
import io
import json
import base64
from streamlit_webrtc import AudioProcessorBase, webrtc

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


# Set up audio stream for recording
FORMAT = pyaudio.paInt16
CHANNELS = 1
CHUNK = FRAMES_PER_BUFFER


        

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

# create a class to write audio data to
class AudioRecorder(AudioProcessorBase):
    def __init__(self, chunk_size: int, channels: int, sample_rate: int):
        self.chunk_size = chunk_size
        self.channels = channels
        self.sample_rate = sample_rate
        self.frames = []

    def on_chunk(self, chunk: av.AudioFrame) -> None:
        # Convert audio chunk to numpy array
        audio = np.frombuffer(chunk.to_ndarray(), np.int16)

        # Append audio chunk to list of audio frames
        self.frames.append(audio)

    def get_audio(self)-> np.ndarray:
        # Convert list of audio frames to single numpy array
        audio_np = np.concatenate(self.frames)

        # Reshape audio array to 2D array with shape (-1, channels)
        #audio_np = audio_np.reshape((-1, self.channels))

        return audio_np

# Define a function that expects a bytes-like object as input
def process_data(data):
    # Check if the input data is a bytes-like object
    if not isinstance(data, (bytes, bytearray)):
        raise TypeError("Input data must be a bytes-like object")
        


    
# Function to record audio for specified duration
async def record_audio(duration: int, chunk_size: int, channels: int, sample_rate: int) -> np.ndarray:
    recorder = AudioRecorder(chunk_size, channels, sample_rate)
    t0 = time.time()

    # Start recording audio
    #webrtc_ctx = webrtc(key="audio", audio_receiver_processor_factory=recorder, sendback_audio=False)
    
    # Create a webrtc component with audio input
    #webrtc_ctx = webrtc.Streamer(audio=True)
    # Create a webrtc component with audio input and the specified settings
    webrtc_ctx = webrtc.Streamer(
        audio=True,
        audio_receiver_size=2048,  # Number of audio frames to buffer
        max_audio_bitrate=128,     # Maximum audio bitrate in kbps
        audio_processor_factory=webrtc.MediaStreamAudioProcessorFactory(
            input_sample_rate=RATE,
            output_sample_rate=RATE,
            channels=CHANNELS,
            chunk_size=CHUNK,
            format="S16LE"  # Audio format (signed 16-bit little-endian)
        )
    )

    # Check if the webrtc component is capturing audio
    if webrtc_ctx.audio_receiver:
        st.write("Audio is being captured.")
    
    # Keep the webrtc component running and process incoming audio data
    # Record audio for specified duration
    # Record for 20 seconds
    start_time = time.time()
    while (time.time() - start_time) < duration:
        # Wait for new audio data to arrive
        audio_np = webrtc_ctx.audio_receiver.get_frames()

    # Stop recording audio
    #webrtc_ctx.audio_receiver_processor = None

    # Get recorded audio data
    #audio_np = await recorder.get_audio()
    print("audio_np", audio_np)
    
    # Convert audio data to byte stream using WAV encoding
    #audio_bytes = audio_np.to_wav()
    
    # Convert the numpy array to a list
    #audio_data_list = audio_np.tolist()
    data = base64.b64encode(data_np).decode("utf-8")

    # Create a dictionary containing the audio data and any other metadata
    data_dict = {"audio_data": str(data), "sample_rate": RATE}

    # Encode the dictionary as a UTF-8 JSON string
    #json_str = json.dumps(data_dict).encode('utf-8')
    json_str = json.dumps(data_dict)

    return json_str
    

# the "save_audio_file" will not be used.
async def save_audio_file(audio_data, filename):
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b"".join(audio_data))
        wf.close()
        

async def send_audio_train(json_str):
    async with websockets.connect("ws://35.87.244.144:8000/train") as ws:
        try:
            #await ws.send(audio_data.tobytes())
            await ws.send(json_str)
            
            st.write("Recorded audio is sent to the EC2 instance.")
            await ws.recv()
            st.write("Training completed")
        finally:
            ws.close()

st.markdown('### :white[Record and Train]')
st.markdown(':violet[Press "*Record*" and Repeat Your Custom Keyword for 30 Seconds.]')
st.markdown(':violet[Audio Will be Sent to EC2 Instance to Fine-Tune the Model]')
duration = st.slider("Select recording duration (in seconds)", 20, 60, 25) # minimum of 20 second recording
if st.button('Record 🎙️ and Train 🛠️'):
    st.spinner(text="In progress...")
    json_str = record_audio(duration, CHUNK, CHANNELS, RATE)
    asyncio.run(send_audio_train(json_str))
    
#===========================================================
async def predict_audio():
    async with websockets.connect("ws://35.87.244.144:8000/inference") as ws:
        while st.session_state['run']:
            # capture real-time audio as bytes
            duration =1
            json_str = record_audio(duration, CHUNK, CHANNELS, RATE)
            #await ws.send(frames.tobytes())
            await ws.send(json_str)
            
            #await ws.send(frames.tolist())
            # wait for prediction result
            result = await ws.recv()
            # display prediction result
            st.write(result)


        ws.close()   

    
st.markdown('### :white[Real-Time Evaluation]')
st.markdown(':violet[Press "*Start Inference*" to Test the Fine-Tuned Custom Keyword Spotting Model in Real-Time]')
if st.button('Stop Inference ⏹️'):
    st.session_state['run'] = False
    
if st.button('Start Inference ▶️'):
    st.session_state['run'] = True
    asyncio.run(predict_audio())
    


#option = st.sidebar.selectbox("Select an option", ["Train", "Predict"])

    


