# # start by pulling the python image
FROM python:3.8

RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6

RUN pip install opencv-python-headless

# copy the requirements file into the image
COPY ./requirements.txt /app/requirements.txt

# switch working directory
WORKDIR /app

# install the dependencies and packages in the requirements file
RUN pip install -r requirements.txt


# copy every content from the local file to the image
COPY . /app

# configure the container to run in an executed manner
ENTRYPOINT [ "python" ]

CMD ["app.py" ]

# Start with the CUDA base image
# FROM nvidia/cuda:11.4.1-cudnn8-runtime-ubuntu20.04

# # Install required dependencies
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     libglib2.0-0 \
#     libsm6 \
#     libxrender1 \
#     libxext6 \
#     python3-dev \
#     python3-pip 
#     # libcudnn8

# # Set environment variables for CUDA
# ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"
# ENV PATH="/usr/local/cuda/bin:${PATH}"

# # Install CUDA-aware OpenCV and other dependencies
# RUN pip3 install opencv-python-headless tensorflow

# # Copy the requirements file into the image
# COPY ./requirements.txt /app/requirements.txt

# # Switch working directory
# WORKDIR /app

# # Install the dependencies and packages in the requirements file
# RUN pip3 install -r requirements.txt

# # Copy every content from the local file to the image
# COPY . /app

# # Configure the container to run in an executed manner
# ENTRYPOINT ["python3"]
# CMD ["app.py"]


# FROM nvidia/cuda:11.6.0-base-ubuntu20.04

# RUN apt-get update && apt-get install -y \
#     libglib2.0-0 \
#     libsm6 \
#     libxrender1 \
#     libxext6 \
#     cuda

# RUN pip install opencv-python-headless --headless

# # add NVIDIA Container Toolkit
# RUN curl https://get.docker.com | sh \
#     && systemctl --now enable docker
# RUN distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
#     && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | apt-key add - \
#     && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | tee /etc/apt/sources.list.d/nvidia-docker.list
# RUN apt-get update && apt-get install -y nvidia-docker2

# # copy the requirements file into the image
# COPY ./requirements.txt /app/requirements.txt

# # switch working directory
# WORKDIR /app

# # install the dependencies and packages in the requirements file
# RUN pip install -r requirements.txt

# # copy every content from the local file to the image
# COPY . /app

# # configure the container to run in an executed manner with NVIDIA Container Toolkit
# ENTRYPOINT [ "nvidia-docker", "run", "-it", "--rm", "--ipc=host", "--network=host", "--device=/dev/video0" ]

# CMD ["app.py"]