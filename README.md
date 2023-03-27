# Real-time Multilingual Custom Keyword Spotting via Few Shot Learning
- Proposing a lightweight few-shot keyword spotting (FS-KWS) for personalization of keyword spotting or wake-word detection running in real-time on an edge device.
- The customization is achieved by recording audio from the user for less than 30 seconds, then segmenting the speaker's audio to 1-second speech audio files via a deep learning model [1]. Segmented audio files were then used to fine-tune and customize an efficientnet-B0-based multilingual keyword spotting model through few-shot learning. The baseline model that we used in this study is based on [Harvard-edge](https://github.com/harvard-edge/multilingual_kws)[2] work.
- Our model operates on short audio chunks (1 second) but at a much higher temporal resolution (every 25 ms).
- Average processing time for each 1-second audio chunk observed to be 60 ms.


## Solution Architecture
![Solution Architecture](Documents/MLE11_KWS_Solution_Architecture2.jpg)

## Component Setup - step by step
**1.** Clone the Project 
```
git clone https://github.com/nasim-alamdari/RealTime-Custom-Keyword-Spotting.git
cd RealTime-Custom-Keyword-Spotting
```

**2.** Import and install relevant libraries to your Python project. 
```
conda create --name kws
conda activate kws
pip install -r requirement.txt
```
**3.** Regarding installing Pyaudio on M1 Mac or EC2 :
```
# for M1 Mac:
brew install portaudio
pip install PyAudio

# for EC2 Ubuntu:
sudo apt install libasound-dev portaudio19-dev libportaudio2 libportaudiocpp0 ffmpeg libav-tools
pip install PyAudio
```
**4. Enjoy real-time custom keyword spotting inside your streamlit app! 🎈**
```
cd Code
streamlit run streamlit_rltime_app.py
```


## Deployment via Streamlit:**

![streamlit App](Images/streamlit_scrnshot.png)




#### References
1. End-to-End Speaker Segmentation for Overlap-aware Resegmentation, Proc. Interspeech 2021, [link to paper.](https://arxiv.org/pdf/2104.04045.pdf)
2. Few-Shot Keyword Spotting in Any Language, Proc. Interspeech 2021, [link to paper.] (https://www.isca-speech.org/archive/pdfs/interspeech_2021/mazumder21_interspeech.pdf)


Feel free to reach out to us in case you have any questions! <br>
Pls consider leaving a `star` ☆ with this repository to show your support.

#### Contributors: 
[Nasim Alamdari](https://www.linkedin.com/in/nasim-alamdari/) and [Christos Magganas](https://www.linkedin.com/in/christos-magganas/)








