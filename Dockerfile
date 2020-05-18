FROM python:latest
COPY ./tokenizer /server
WORKDIR /server
EXPOSE 8000
RUN pip3 install -r requirements.txt
CMD python ./run.py
