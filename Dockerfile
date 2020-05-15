FROM tensorflow/serving
COPY ./SavedModels/DocNet /models/docnet 
ENV MODEL_NAME docnet

