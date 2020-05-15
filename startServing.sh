#!/bin/bash

sudo docker run -p 8501:8501 \
  --mount type=bind,source=/home/daniel/Desktop/DocNet/SavedModels/DocNet,target=/models/docnet \
  -e MODEL_NAME=docnet -t tensorflow/serving
