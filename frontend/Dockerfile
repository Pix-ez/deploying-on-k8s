# FROM node:22-alpine3.18

# RUN addgroup app  && adduser -S -G app app

# USER app

# WORKDIR /app

# COPY package*.json ./

# USER root

# RUN chown -R app:app .

# USER app

# RUN npm install 

# COPY . .

# EXPOSE 5173

# CMD npm run dev



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
