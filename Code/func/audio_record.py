
"""
installation on Mac M1:
1. brew install portaudio
2. python -m pip install --global-option='build_ext' --global-option='-I/opt/homebrew/Cellar/portaudio/19.7.0/include' --global-option='-L/opt/homebrew/Cellar/portaudio/19.7.0/lib' pyaudio
"""

import os
import pyaudio
#import torchaudio
import numpy as np
from scipy.io.wavfile import write, read
import struct
import IPython
    
def record (
    duration: int,
    record_name: str,
    record_save_path: os.PathLike,
    ):

    #=====================================
    # Open an audio stream from the microphone with PyAudio

    INPUT_BLOCK_TIME = 0.025
    sample_rate = 16000
    CHUNK = int(sample_rate*INPUT_BLOCK_TIME) # FRAMES_PER_BUFFER
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    SHORT_NORMALIZE = (1.0/32768.0)

    p = pyaudio.PyAudio()

    #=======================================

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

    device_index = _find_input_device()

    #=======================================
    """player = p.openstream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=sample_rate,
        output=True,
        frames_per_buffer=CHUNK,
    )"""
    # starts recording
    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=sample_rate,
        input=True,
        input_device_index = device_index,
        frames_per_buffer=CHUNK,
    )

    print("Start recording...")
    frames = []
    cnt = 0

    for i in range(int(duration*sample_rate/CHUNK)): #do this for 10 seconds
        # to play back
        #player.write(np.frombuffer(stream.read(CHUNK),dtype=np.int16),CHUNK)
        #block = np.frombuffer(stream.read(CHUNK),dtype=np.int16)
        block = stream.read(CHUNK)
        count = len(block)/2
        format = "%dh"%(count)
        shorts = struct.unpack( format, block )
        shorts = np.int16(shorts)
        frames.append(np.squeeze(shorts[:]))

        if cnt==0:
            print(len(block), count, format, len(shorts))

        if cnt%200==0:
            print("...")
        cnt+=1
    
    #=====================================
    wav = np.asarray(frames).flatten()
    
    stream.stop_stream()
    stream.close()
    p.terminate()
    write(os.path.join(record_save_path,record_name),  sample_rate, wav)
    print("Recording Completed")
    fs, wav = read(os.path.join(record_save_path,record_name))
    print("fs and wav len:", fs, len(wav))
    return fs, os.path.join(record_save_path)


#if __name__ == "__main__":
#    record (record_name: str,record_save_path: os.PathLike)



