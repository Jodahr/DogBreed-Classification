time curl -X POST -F image=@shepherd.jpg 'http://localhost:5000/predict'
sudo docker build -t flask-sample-one:latest .
sudo docker run -d -p 5000:5000 flask-sample-one

gunicorn -b 127.0.0.1:5000 -w 4 rest_api:app

