# Nasim Alamdari
# last Update March 2023

"""
# for speechbrain
!pip install -qq torch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 torchtext==0.12.0
!pip install -qq speechbrain==0.5.12

# pyannote.audio
!pip install -qq pyannote.audio

# for visualization purposes
!pip install -qq ipython==7.34.0
"""

# Call speaker-segmentation model from pyannote:
from pyannote.audio import Pipeline
from scipy.io import wavfile
import matplotlib.pyplot as plt
import numpy as np
import sox
import os
from pathlib import Path
import subprocess
from dotenv import load_dotenv
load_dotenv()
HF_TOKEN = os.getenv('HF_TOKEN')

def segment (
    keyword: str,
    recording_path: os.PathLike,
    fs: int):

    #keyword = "heynasim"
    #recording_path = './content/target_kw/recording/heynasim/'

    wav_path = os.path.join(recording_path, keyword+'.wav')
    #print(wav_path)
    #fs, wav = wavfile.read(wav_path)
    
    pipeline = Pipeline.from_pretrained('pyannote/speaker-segmentation',
                        use_auth_token=HF_TOKEN) # you can obtrain you HF_TOKEN from huggingface, security-tokens
    output = pipeline(wav_path)
    
    def _plot_out(wav, output):
        fs, wav = wavfile.read(wav_path)
        plt.figure(figsize = (10,3))
        plt.plot(wav)
        ymax = max(wav)

        for turn, _, speaker in output.itertracks(yield_label=True):
            # speaker speaks between turn.start and turn.end
            #print(turn.end- turn.start )
            plt.plot([ turn.start*fs, turn.end*fs - 1], [ymax * 1.1, ymax * 1.1], color = 'orange')

        plt.xlabel('sample')
        plt.grid()
        
    #_plot_out(wav,output)    
    extractions = os.path.join(recording_path, "extractions")
    if not os.path.exists(extractions):
        os.makedirs(extractions)

    #Convert wavform to correct format in WAV:
    #!ffmpeg -hide_banner -loglevel error -y -i wav_path -acodec pcm_s16le -ac 1 -ar 16000 wav_path
    subprocess.run('ffmpeg -hide_banner -loglevel error -y -i wav_path -acodec pcm_s16le -ac 1 -ar 16000 wav_path', shell=True)
    
    cnt = 0
    for turn, _, speaker in output.itertracks(yield_label=True):
        # speaker speaks between turn.start and turn.end
        cnt+=1
        time_ms_st = turn.start*fs # convert to sec
        time_ms_en = turn.end*fs # convert to sec


        turn_st_flr = np.round(turn.start*1000)/1000
        dest_wav = os.path.join(extractions,str(f"{cnt:03d}_{keyword}_detection_{turn_st_flr}sec.wav"))

        # make sure audio segment is 1sec
        if (time_ms_en - time_ms_st +1) < fs:
            append_len_ms = fs - (time_ms_en - time_ms_st +1)
            append_len_s =  append_len_ms/fs # convert to sec
        #print(turn.start, turn.end, append_len_s)

        transformer = sox.Transformer()
        transformer.convert(samplerate=16000)  
        transformer.trim(turn.start, turn.end + append_len_s)
        transformer.build(wav_path, dest_wav)
    #print("Speaker Segmentation Completed.")
    return extractions


    
if __name__ == "__main__":
    keyword = "heynasim"
    recording_path = './content/target_kw/recording/heynasim/'
    fs = 16000
    extractions = segment (keyword,recording_path,fs)





