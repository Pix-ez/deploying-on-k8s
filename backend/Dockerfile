FROM python:3.9.19-bookworm



WORKDIR /app



# create the app user
RUN addgroup --system app && adduser --system --group app




RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  libgl1-mesa-glx python3-opencv -y 

# chown all the files 
RUN chown -R app:app .

COPY requirements.txt ./

# lint
RUN pip install --upgrade pip

COPY . /app

RUN mkdir uploads && chmod 777 uploads

#install pytorch
RUN pip install torch==2.0.0  --index-url https://download.pytorch.org/whl/cpu

#install requirements
RUN pip install -r requirements.txt --no-cache-dir




EXPOSE 5002

# change to the app user
USER app

CMD python app.py