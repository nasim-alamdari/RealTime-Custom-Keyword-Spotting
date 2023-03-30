# Nasim Alamdari
# last Update March 2023

# Using speech enhancement via HuggingFace SpeechBrain:
from speechbrain.pretrained import SepformerSeparation as separator
import torchaudio
from scipy.io.wavfile import write, read
import subprocess
import os

def se_model(wav_path):

    # Apply speech enhancement
    se_model = separator.from_hparams(source="speechbrain/sepformer-wham-enhancement", savedir='./pretrained_models/sepformer-wham-enhancement')
    est_sources = se_model.separate_file(wav_path) 
    torchaudio.save(wav_path, est_sources[:, :, 0].detach().cpu(), 8000)
    
    # Run ffmpeg command to resample audio and make sure audio is in PCM format
    dir_path, filename = os.path.split(wav_path)
    input_file = wav_path
    output_file = os.path.join(dir_path , "test.wav")
    subprocess.run(['ffmpeg', '-i', input_file, '-ar', '16000', '-acodec', 'pcm_s16le', '-y', output_file], check=True)
    os.remove(input_file) # Delete the original file
    os.rename(output_file, input_file) # Rename the output file to the original filename
    print("Speech enhancement completed")
    
    fs, wav = read(wav_path)
    print("fs and wav len:", fs, len(wav))
    return fs
    
    
if __name__ == "__main__":
    
    wav_path = './content/target_kw/recording/heybob/heybob.wav'
    fs = se_model(wav_path)





