# FS-KWS: Personalized-Key-Word-Spotting-via-Few-Shot-Learning
a low-latency few-shot keyword spotting (FS-KWS) for personalization of key word spotting  or wake word detection running in real-time on an edge device.

Contributors: **Nasim Alamdari and Christos Magganas**

Solution Architecture
![alt text](https://drive.google.com/file/d/1-LrUfbBSF1NQKUMHUYhqmB4JvlnXeO8D/view?usp=sharing)


Training and Inference Steps:
![alt text](https://drive.google.com/file/d/1DrQ5khw5q7iIX9OeNztKqrgv9yT8Uzl8/view?usp=sharing)


**Deployment is done with FastAPI and AWS EC2:**
curl -X 'POST' \
  'http://54.213.116.214:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "keyword": "amelia",
  "keyword_dir": "./content/target_kw/amelia/"
}'




