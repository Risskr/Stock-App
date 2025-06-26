# Use the official lightweight Python image.
# https://hub.docker.com/_/python
FROM python:3.11-slim

# Allow statements and log messages to be sent straight to the logs
ENV PYTHONUNBUFFERED True

# Set the working directory in the container
WORKDIR /app

# Copy over the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code (app.py, etc.)
COPY . .

# Command to run the application using a production-grade server
# Gunicorn is automatically installed by the functions-framework
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 app:server
#CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 app_testing:server

