FROM python:3.8

RUN apt-get update && apt-get install -y \
    wget \
    git \
    portaudio19-dev \
    python3-pyaudio \
    libcairo2-dev \
    sox \
    ffmpeg \
    # libsndfile1 \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda && \
    rm Miniconda3-latest-Linux-x86_64.sh && \
    /opt/conda/bin/conda create -n kws python=3.8 -y && \
    echo "conda activate kws" >> ~/.bashrc

ENV PATH="/opt/conda/envs/kws/bin:$PATH"

RUN git clone -b cool https://github.com/nasim-alamdari/RealTime-Custom-Keyword-Spotting.git && \
    cd RealTime-Custom-Keyword-Spotting && \
    pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install PyAudio torchvision==0.12.0 torchaudio==0.11.0 pyannote.audio streamlit uvicorn websockets fastapi && \
    # possible incompatible versions of llvmlite, numba, numpy
    # pip uninstall llvmlite numba numpy && \
    cd Code

CMD ["streamlit", "run", "RealTime-Custom-Keyword-Spotting/Code/streamlit_app.py"]
