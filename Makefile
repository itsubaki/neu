SHELL := /bin/bash

update:
	go get -u
	go mod tidy

test:
	go test -cover $(shell go list ./... | grep -v /vendor/ | grep -v /build/) -v -coverprofile=coverage.txt -covermode=atomic
	go tool cover -html=coverage.txt -o coverage.html

.PHONY: mnist
mnist:
	curl -s -o testdata/train-images-idx3-ubyte.gz http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
	curl -s -o testdata/train-labels-idx1-ubyte.gz http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
	curl -s -o testdata/t10k-images-idx3-ubyte.gz  http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
	curl -s -o testdata/t10k-labels-idx1-ubyte.gz  http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
