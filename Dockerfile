# Use an official Python runtime as a parent image
FROM python:3.6-slim-stretch

# Set the working directory to /app
WORKDIR /app

# Copy the requirements.txt into the container at /app (they change rarely and installation will be cached)
COPY requirements.txt /app

# Installing pytorch-cpu, because it takes less space and cuda is not required for this task
RUN pip3 install -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . /app

# Run main.py when the container launches
ENTRYPOINT ["python", "main.py"]