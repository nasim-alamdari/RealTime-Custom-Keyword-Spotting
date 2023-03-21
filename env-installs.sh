conda update -n base -c defaults conda
pip install -r requirements.txt

conda install -c apple tensorflow-deps
pip install tensorflow-macos

pip install torch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0
pip install pyannote.audio
conda install -c conda-forge libsndfile

brew install portaudio
python -m pip install --global-option='build_ext' --global-option='-I/opt/homebrew/Cellar/portaudio/19.7.0/include' --global-option='-L/opt/homebrew/Cellar/portaudio/19.7.0/lib' pyaudio
