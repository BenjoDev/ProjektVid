build:
	docker image build -t flask-server .

run:
	docker run -p 5000:5000 -d flask-server
