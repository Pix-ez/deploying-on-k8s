
# Deploying ml app on kubernetes ✨

First setting up indivisual docker image for frontend and flask-server 



## First setting up indivisual docker image for frontend and flask-server


setting docker file:    
for react app we are first building bundled js files and then serving on nginx in same image (😅 It's not necessary to do this we can host this on vercel aslo)

```dockerfile
  # Stage 0, "build-stage", based on Node.js, to build and compile the frontend
    FROM node:22-alpine3.18 as build-stage

    WORKDIR /app

    COPY package*.json /app/

    RUN npm install 

    COPY ./ /app/

    RUN npm run build

    # Stage 1, based on Nginx, to have only the compiled app,
    # ready for production with Nginx
    FROM nginx:stable-perl

    COPY --from=build-stage /app/dist/ /usr/share/nginx/html

    COPY ./nginx.conf /etc/nginx/conf.d/default.conf

    EXPOSE 80

```

setting up flask-server installing requirements

```dockerfile
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
```
creating config files for kubernets deployment in yaml-

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: flask-server-deployment
  labels:
    app: flask-server
spec:
  replicas: 1
  selector:
    matchLabels:
      app: flask-server
  template:
    metadata:
      labels:
        app: flask-server
    spec:
      containers:
      - name: flask-server
        image: rahult046/my-app:flask-server
        ports:
        - containerPort: 5002
        
---
apiVersion: v1
kind: Service
metadata:
  name: flask-server-service
spec:
  selector:
    app: flask-server
  ports:
    - protocol: TCP
      port: 5002
      targetPort: 5002
```

#### Here I'm using minikube (Minikube creates a single node cluster inside a VM on your System. It is good for beginners to learn Kubernetes since you don't have to create a master and a minimum of one worker node to create a cluster and still, practice basic Kubernetes functions and can also install the Kubernetes dashboard.)

```bash
minikube start --driver=docker

kubectl create -f flask.yml
```

