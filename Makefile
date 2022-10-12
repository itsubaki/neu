SHELL := /bin/bash

update:
	go get -u
	go mod tidy

test:
	go test -cover $(shell go list ./... | grep -v /vendor/ | grep -v /build/) -v -coverprofile=coverage.txt -covermode=atomic
	go tool cover -html=coverage.txt -o coverage.html

mnist:
	curl -s -H 'User-Agent: Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:47.0) Gecko/20100101 Firefox/47.0' -o testdata/train-images-idx3-ubyte.gz http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
	curl -s -H 'User-Agent: Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:47.0) Gecko/20100101 Firefox/47.0' -o testdata/train-labels-idx1-ubyte.gz http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
	curl -s -H 'User-Agent: Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:47.0) Gecko/20100101 Firefox/47.0' -o testdata/t10k-images-idx3-ubyte.gz  http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
	curl -s -H 'User-Agent: Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:47.0) Gecko/20100101 Firefox/47.0' -o testdata/t10k-labels-idx1-ubyte.gz  http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
