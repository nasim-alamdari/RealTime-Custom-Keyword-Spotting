FROM python:3.8

# Set the working directory to the root of your project inside the container.
WORKDIR .


#Copy the file with the requirements to the (current/root) directory.
COPY requirements.txt .

# Install appropriate dependencies.
RUN pip install --no-cache-dir -U -r  requirements.txt

COPY ./Code

EXPOSE 8501
CMD ["streamlit", "run", "streamlit_rltime_app.py"]

