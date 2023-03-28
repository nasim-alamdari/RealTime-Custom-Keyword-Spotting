wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
# export PATH="/home/ubuntu/miniconda3/bin:$PATH"

conda create -n kws python=3.8 -y
conda activate kws
git clone https://github.com/nasim-alamdari/RealTime-Custom-Keyword-Spotting.git
cd RealTime-Custom-Keyword-Spotting
sudo apt-get update # sudo apt-get update && sudo apt-get upgrade -y
sudo apt install python3-pip -y
pip install --upgrade pip

sudo apt-get install portaudio19-dev python3-pyaudio libcairo2-dev -y
# sudo apt-get install tmux
pip install PyAudio
pip install pyannote.audio
pip install torchvision==0.12.0
pip install torchaudio==0.11.0
sudo apt-get -y install sox
# sudo apt install ffmpeg
pip install streamlit uvicorn
pip install websockets
# pip install streamlit_webrtc
pip install fastapi

pip install -r requirements.txt
# export PATH=$PATH:~/.local/bin

cd Code
streamlit run streamlit_app.py