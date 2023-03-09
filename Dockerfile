# Set the base image to use for building the Docker image
FROM python:3.8-slim-buster

# Set the working directory
WORKDIR /app

# Copy the requirements file into the container and install the dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy the rest of the app into the container
COPY . .

# Expose the port that the Flask app will run on
EXPOSE 5000

# Start the Flask app
CMD ["flask", "run"]

