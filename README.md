# FS-KWS: Personalized Multilingual Keyword Spotting via Few Shot Learning
a low-latency few-shot keyword spotting (FS-KWS) for personalization of key word spotting  or wake word detection running in real-time on an edge device.

Contributors: **Nasim Alamdari and Christos Magganas**

Solution Architecture
![Solution Architecture](Documents/MLE11_KWS_Solution_Architecture2.jpg)


**Deployment is done with FastAPI and AWS EC2:**
curl -X 'POST' \
  'http://54.213.116.214:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "keyword": "amelia",
  "keyword_dir": "./content/target_kw/amelia/"
}'




