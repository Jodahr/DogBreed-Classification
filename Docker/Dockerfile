FROM ubuntu:latest
MAINTAINER Marcel Rothering "marcelrothering@fastmail.fm"
RUN apt-get update -y
RUN apt-get install -y python3-pip python3-dev build-essential
COPY . /app
WORKDIR /app
RUN pip3 install -r requirements.txt
ENTRYPOINT ["python3"]
CMD ["rest_api.py"]