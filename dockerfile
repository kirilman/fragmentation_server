FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime
# RUN sudo apt install python3.9
# RUN mkdir /app
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y 
RUN apt-get install wget -y
RUN apt-get install unzip -y 
WORKDIR /app
COPY . .
RUN pip install -r ./app/requirements.txt
RUN wget https://github.com/kirilman/ASBEST_VEINS_LABELING/archive/refs/heads/asbest.zip
RUN ls -l   
RUN unzip asbest
RUN cd ./ASBEST_VEINS_LABELING-asbest && pip install --no-cache-dir -e .
RUN cd ../
CMD python3 ./app/main.py
EXPOSE 8787