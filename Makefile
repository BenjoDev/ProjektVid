build:
	docker image build -t flask-server .

run:
# docker run --gpus all -p 5000:5000 -d flask-server

	docker run -p 5000:5000 -d flask-server	
# docker run -e CUDA_VISIBLE_DEVICES=-1 -it tensorflow/tensorflow:latest
