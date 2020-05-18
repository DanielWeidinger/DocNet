FROM python:latest
COPY ./faiss_service /server
WORKDIR /server
RUN pip install -r requirements.txt
EXPOSE 7000
CMD python run.py
