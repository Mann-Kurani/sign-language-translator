# Use an official Python runtime as a parent image
FROM python:3.8-slim

RUN apt-get update && apt-get install -y libgl1-mesa-glx
RUN apt-get update && apt-get install -y libglib2.0-0

# Copy the checkpoint.pth and signlanguage.onnx files into the container at /app
COPY checkpoint.pth signlanguage.onnx /app/

# Set the working directory in the container
WORKDIR /app

# Install any needed dependencies specified in requirements.txt
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the Python scripts into the container at /app
COPY step_2_dataset.py step_3_train.py step_4_evaluate.py step_5_camera.py /app/

# Define environment variable
ENV PYTHONUNBUFFERED=TRUE

EXPOSE 80
# Run step_5_camera.py when the container launches
CMD ["python", "step_5_camera.py"]
