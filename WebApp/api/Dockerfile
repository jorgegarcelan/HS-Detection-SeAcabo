# Use an official Python runtime as a parent image
FROM python:3.11.7

# Set the working directory in the container
WORKDIR /nlp_api

# Copy the current directory contents into the container at /nlp_api
COPY . /nlp_api

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Run flask when the container launches
CMD [ "flask", "run", "--host=0.0.0.0", "--port=8000" ]